"""
meridian.py - Monte Carlo equity-based No-Limit Hold'em bot.
Fully self-contained: no imports from engine.py.

Strategy summary
-----------------
On every street (including preflop) this bot estimates its equity by
simulating many random completions of the hand: it deals random hole
cards to each live opponent, fills out the remaining board randomly,
and scores the showdown with a hand evaluator built into this file.
The fraction of simulated showdowns it wins (splitting ties fairly)
is its equity estimate. There is no separate preflop hand chart --
the simulation naturally values pairs, suited/connected cards, and
multiway dilution because those are exactly what more opponents /
random run-outs capture.

Decisions are then made by comparing equity to pot odds:
  - Facing no bet: check most of the time, bet for value with strong
    equity, and occasionally bet/semi-bluff with medium equity.
  - Facing a bet: fold when equity is well below what the pot is
    laying, call when it's close, raise for value when it's well
    ahead, with a small amount of randomized bluffing/hero-calling
    so the bot isn't perfectly exploitable.

Because there are no blinds, "pot == 0" (nobody has ever bet) is a
special case -- bet sizing there falls back to a stack percentage
instead of a pot percentage.

All raise/bet amounts are clamped against a locally-reconstructed
picture of the betting round (built by replaying `action_history`)
so that every action this bot returns is guaranteed legal under the
engine's `_action_is_legal` check. If anything unexpected happens,
the bot falls back to the safest legal action instead of risking an
auto-forfeit.
"""

import itertools
import random
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Local copies of engine constants / card helpers (no engine import needed)
# ---------------------------------------------------------------------------
STATUS_FOLDED = "folded"

SUITS = ["H", "D", "C", "S"]
RANKS = list(range(2, 15))  # 2..14, Ace = 14


def _build_deck():
    return [(s, r) for s in SUITS for r in RANKS]


def _straight_high(ranks):
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    unique = sorted(set(unique), reverse=True)
    for i in range(len(unique) - 4):
        window = unique[i : i + 5]
        if window[0] - window[4] == 4:
            return window[0]
    return None


def _evaluate_five(cards):
    """Comparable tuple for five cards; higher is better."""
    ranks = sorted((c[1] for c in cards), reverse=True)
    suits = [c[0] for c in cards]
    counts = Counter(ranks)
    by_freq = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if is_flush and straight_high:
        return (8, straight_high)
    if by_freq[0][1] == 4:
        return (6, by_freq[0][0], by_freq[1][0])
    if by_freq[0][1] == 3 and by_freq[1][1] == 2:
        return (5, by_freq[0][0], by_freq[1][0])
    if is_flush:
        return (4, *ranks)
    if straight_high:
        return (3, straight_high)
    if by_freq[0][1] == 3:
        trips = by_freq[0][0]
        kickers = [r for r in ranks if r != trips]
        return (2, trips, *kickers)
    if by_freq[0][1] == 2 and by_freq[1][1] == 2:
        hi, lo = max(by_freq[0][0], by_freq[1][0]), min(by_freq[0][0], by_freq[1][0])
        kicker = [r for r in ranks if r not in (hi, lo)][0]
        return (1, hi, lo, kicker)
    if by_freq[0][1] == 2:
        pair = by_freq[0][0]
        kickers = [r for r in ranks if r != pair]
        return (0, pair, *kickers)
    return (-1, *ranks)


def _evaluate_best_hand(hole, board):
    """Best 5-card hand from hole + board cards."""
    all_cards = list(hole) + list(board)
    best = None
    for combo in itertools.combinations(all_cards, 5):
        score = _evaluate_five(combo)
        if best is None or score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
BASE_TIME_BUDGET = 1.55          # seconds; leaves margin under the 2.0s cap
MAX_SIMS = 3000
MIN_SIMS = 40

RAISE_BIG_LOW, RAISE_BIG_HIGH = 0.6, 1.0     # x pot, value raise
RAISE_MED_LOW, RAISE_MED_HIGH = 0.35, 0.6    # x pot, thinner raise
BET_STRONG_LOW, BET_STRONG_HIGH = 0.55, 0.85  # x pot, value bet
BET_SEMI_LOW, BET_SEMI_HIGH = 0.3, 0.5        # x pot, semi-bluff / thin bet
OPEN_STACK_LOW, OPEN_STACK_HIGH = 0.015, 0.035  # x stack, when pot == 0


# ---------------------------------------------------------------------------
# Betting-round reconstruction (view doesn't expose street_wager directly)
# ---------------------------------------------------------------------------
def _reconstruct_street_wagers(view):
    """Replay this street's action_history to recover each player's
    current street wager and the current bet level, so we can compute
    an exact, always-legal all-in raise target."""
    wagers = {p: 0 for p in view.seat_order}
    current_level = 0
    for player, action in view.action_history:
        kind = action[0]
        if kind in ("fold", "check"):
            continue
        if kind == "call":
            wagers[player] = current_level
        elif kind in ("bet", "raise"):
            wagers[player] = action[1]
            current_level = action[1]
    return wagers, current_level


def _safe_raise_to(view, desired_to, wagers):
    """Clamp a desired raise target into the legal [min_raise_to, max_to]
    window. Returns None if no legal raise exists right now."""
    min_to = view.min_raise_to
    if min_to is None:
        return None
    my_wager = wagers.get(view.your_name, 0)
    max_to = my_wager + view.your_stack
    if min_to > max_to:
        return None
    to_total = int(max(min_to, min(desired_to, max_to)))
    return ("raise", to_total)


def _safe_bet(view, desired_amt):
    amt = int(max(1, min(desired_amt, view.your_stack)))
    return ("bet", amt)


# ---------------------------------------------------------------------------
# Equity estimation
# ---------------------------------------------------------------------------
def _live_opponents(view):
    return [
        p for p in view.seat_order
        if p != view.your_name and view.player_status.get(p) != STATUS_FOLDED
    ]


def _estimate_equity(hole, board, num_opponents, time_budget):
    """Monte Carlo win probability for `hole` given `board`, against
    `num_opponents` random live hands, integrating over random
    completions of the remaining board (i.e. full showdown equity)."""
    if num_opponents <= 0:
        return 1.0

    deck = _build_deck()
    known = set(hole) | set(board)
    remaining = [c for c in deck if c not in known]

    needed_board = 5 - len(board)
    draw_size = num_opponents * 2 + needed_board
    if draw_size > len(remaining):
        # Should not happen at a real table, but stay safe.
        return 0.5

    wins = 0.0
    trials = 0
    deadline = time.time() + time_budget

    while trials < MAX_SIMS and (trials < MIN_SIMS or time.time() < deadline):
        trials += 1
        sample = random.sample(remaining, draw_size)
        opp_hands = [sample[i * 2:i * 2 + 2] for i in range(num_opponents)]
        board_fill = sample[num_opponents * 2:num_opponents * 2 + needed_board]
        full_board = board + board_fill

        my_score = _evaluate_best_hand(hole, full_board)
        best_opp = None
        tied_opps = 0
        for oh in opp_hands:
            score = _evaluate_best_hand(oh, full_board)
            if best_opp is None or score > best_opp:
                best_opp = score
                tied_opps = 1
            elif score == best_opp:
                tied_opps += 1

        if best_opp is None or my_score > best_opp:
            wins += 1.0
        elif my_score == best_opp:
            wins += 1.0 / (tied_opps + 1)

    return wins / trials if trials else 0.5


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------
def _decide(view):
    hole = view.your_hole_cards
    board = view.community_cards
    to_call = view.amount_to_call
    pot = view.pot
    stack = view.your_stack

    opponents = _live_opponents(view)
    num_opp = len(opponents)
    if num_opp == 0:
        return ("check",) if to_call == 0 else ("call",)

    wagers, _ = _reconstruct_street_wagers(view)
    my_wager = wagers.get(view.your_name, 0)

    time_budget = max(0.35, BASE_TIME_BUDGET - 0.08 * max(0, num_opp - 2))
    equity = _estimate_equity(hole, board, num_opp, time_budget)

    bluff_roll = random.random()

    # -------------------------------------------------------------
    # Facing no bet: check or bet
    # -------------------------------------------------------------
    if to_call == 0:
        if equity > 0.65:
            if pot > 0:
                amt = pot * random.uniform(BET_STRONG_LOW, BET_STRONG_HIGH)
            else:
                amt = stack * random.uniform(OPEN_STACK_LOW, OPEN_STACK_HIGH)
            return _safe_bet(view, amt)

        if equity > 0.5 and bluff_roll < 0.2:
            if pot > 0:
                amt = pot * random.uniform(BET_SEMI_LOW, BET_SEMI_HIGH)
            else:
                amt = stack * random.uniform(OPEN_STACK_LOW, OPEN_STACK_HIGH) * 0.7
            return _safe_bet(view, amt)

        if equity > 0.3 and bluff_roll < 0.06:
            # occasional pure bluff to stay unpredictable
            amt = (pot if pot > 0 else stack * 0.02) * random.uniform(0.4, 0.6)
            return _safe_bet(view, amt)

        return ("check",)

    # -------------------------------------------------------------
    # Facing a bet: fold, call, or raise
    # -------------------------------------------------------------
    pot_odds = to_call / (pot + to_call)

    if to_call >= stack:
        # Calling is already an effective all-in; no raise decision needed.
        if equity >= pot_odds + 0.02 or (equity > 0.35 and bluff_roll < 0.03):
            return ("call",)
        return ("fold",)

    if equity < 0.15:
        return ("fold",)

    if equity < pot_odds - 0.03:
        if bluff_roll < 0.05:
            return ("call",)  # rare hero call for deception
        return ("fold",)

    if equity > pot_odds + 0.25 or equity > 0.78:
        desired_to = my_wager + to_call + pot * random.uniform(RAISE_BIG_LOW, RAISE_BIG_HIGH)
        action = _safe_raise_to(view, desired_to, wagers)
        if action:
            return action
        return ("call",)

    if equity > pot_odds + 0.08 and bluff_roll < 0.35:
        desired_to = my_wager + to_call + pot * random.uniform(RAISE_MED_LOW, RAISE_MED_HIGH)
        action = _safe_raise_to(view, desired_to, wagers)
        if action:
            return action

    return ("call",)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def nextMove(view):
    try:
        return _decide(view)
    except Exception:
        # Defensive fallback: never risk an illegal-action forfeit.
        try:
            if view.amount_to_call == 0:
                return ("check",)
            if view.amount_to_call < view.your_stack * 0.05:
                return ("call",)
            return ("fold",)
        except Exception:
            return ("fold",)

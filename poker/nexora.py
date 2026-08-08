# Apex — elite NLHE bot for BotWars (no blinds, 50k reset stacks).
#
# Core ideas for this ruleset:
#   1. No blinds => fold trash for free; never donate.
#   2. Current field is soft (stations + blind bettors) => value-bet hard,
#      call maniacs wider, punish small opens.
#   3. Fast hand-strength + light Monte Carlo stays under the 2s limit.

import itertools
import random
from collections import Counter, defaultdict

SUITS = ("H", "D", "C", "S")
RANKS = tuple(range(2, 15))
FULL_DECK = [(s, r) for s in SUITS for r in RANKS]

_CHEN = {
    14: 10, 13: 8, 12: 7, 11: 6, 10: 5,
    9: 4.5, 8: 4, 7: 3.5, 6: 3, 5: 2.5, 4: 2, 3: 1.5, 2: 1,
}

# Persistent opponent notes across hands (module survives for the tournament).
_OPP = defaultdict(lambda: {"bets": 0, "folds": 0, "calls": 0, "raises": 0, "sizes": []})


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
    ranks = sorted((c[1] for c in cards), reverse=True)
    suits = [c[0] for c in cards]
    counts = Counter(ranks)
    by_freq = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    sh = _straight_high(ranks)

    if is_flush and sh:
        return (8, sh)
    if by_freq[0][1] == 4:
        return (6, by_freq[0][0], by_freq[1][0])
    if by_freq[0][1] == 3 and by_freq[1][1] == 2:
        return (5, by_freq[0][0], by_freq[1][0])
    if is_flush:
        return (4, *ranks)
    if sh:
        return (3, sh)
    if by_freq[0][1] == 3:
        trips = by_freq[0][0]
        kickers = [r for r in ranks if r != trips]
        return (2, trips, *kickers)
    if by_freq[0][1] == 2 and by_freq[1][1] == 2:
        hi = max(by_freq[0][0], by_freq[1][0])
        lo = min(by_freq[0][0], by_freq[1][0])
        kicker = [r for r in ranks if r not in (hi, lo)][0]
        return (1, hi, lo, kicker)
    if by_freq[0][1] == 2:
        pair = by_freq[0][0]
        kickers = [r for r in ranks if r != pair]
        return (0, pair, *kickers)
    return (-1, *ranks)


def evaluate_best(hole, board):
    cards = list(hole) + list(board)
    best = None
    for combo in itertools.combinations(cards, 5):
        score = _evaluate_five(combo)
        if best is None or score > best:
            best = score
    return best


def chen_score(hole):
    a, b = sorted((hole[0][1], hole[1][1]), reverse=True)
    suited = hole[0][0] == hole[1][0]
    score = _CHEN[a]
    if a == b:
        return max(score * 2, 5)
    score += 2 if suited else 0
    gap = a - b - 1
    if gap == 1:
        score -= 1
    elif gap == 2:
        score -= 2
    elif gap == 3:
        score -= 4
    elif gap >= 4:
        score -= 5
    if gap <= 1 and a < 12 and a - b == 1:
        score += 1
    return score


def hand_category(hole):
    r1, r2 = hole[0][1], hole[1][1]
    hi, lo = max(r1, r2), min(r1, r2)
    suited = hole[0][0] == hole[1][0]
    pair = hi == lo
    chen = chen_score(hole)

    if pair and hi >= 13:
        return "monster"
    if pair and hi == 12:
        return "premium"
    if hi == 14 and lo == 13:
        return "premium"
    if pair and hi >= 10:
        return "strong"
    if hi == 14 and lo >= 12:
        return "strong"
    if hi == 14 and lo == 11 and suited:
        return "strong"
    if chen >= 8:
        return "strong"
    if chen >= 5.5 or (pair and hi >= 5):
        return "playable"
    if suited and hi - lo <= 2 and hi >= 7:
        return "playable"
    if hi == 14 and lo >= 9:
        return "playable"
    if hi >= 12 and lo >= 10:
        return "playable"
    return "trash"


def update_opponent_model(gs):
    """Learn from completed hands (once each)."""
    me = gs.your_name
    if not hasattr(update_opponent_model, "_seen"):
        update_opponent_model._seen = -1
    for hand in gs.hand_history:
        hn = hand["hand_number"]
        if hn <= update_opponent_model._seen:
            continue
        update_opponent_model._seen = hn
        for street_acts in hand.get("actions", {}).values():
            for name, action in street_acts:
                if name == me:
                    continue
                st = _OPP[name]
                k = action[0]
                if k == "fold":
                    st["folds"] += 1
                elif k == "call":
                    st["calls"] += 1
                elif k == "bet":
                    st["bets"] += 1
                    st["sizes"].append(action[1])
                    if len(st["sizes"]) > 40:
                        st["sizes"] = st["sizes"][-40:]
                elif k == "raise":
                    st["raises"] += 1
                    st["sizes"].append(action[1])
                    if len(st["sizes"]) > 40:
                        st["sizes"] = st["sizes"][-40:]


def table_profile(gs):
    """Return (station_score, maniac_score) in [0,1]."""
    update_opponent_model(gs)
    me = gs.your_name
    stations = maniacs = 0
    n = 0
    for name, st in list(_OPP.items()):
        if name == me:
            continue
        acts = st["folds"] + st["calls"] + st["bets"] + st["raises"]
        if acts < 3:
            # Prior: field looks soft
            stations += 0.7
            maniacs += 0.4
            n += 1
            continue
        fold_rate = st["folds"] / acts
        agg = (st["bets"] + st["raises"]) / acts
        call_rate = st["calls"] / acts
        avg_size = sum(st["sizes"]) / len(st["sizes"]) if st["sizes"] else 0
        stations += max(0.0, call_rate - fold_rate + 0.3)
        maniac = agg * 0.7 + (0.4 if avg_size >= 3000 else 0.0)
        maniacs += min(1.0, maniac)
        n += 1
    if n == 0:
        return 0.7, 0.4
    return min(1.0, stations / n), min(1.0, maniacs / n)


def active_opps(gs):
    return [
        n
        for n, st in gs.player_status.items()
        if n != gs.your_name and st in ("active", "all_in")
    ]


def our_street_wager(gs):
    level = 0
    our = 0
    for name, action in gs.action_history:
        k = action[0]
        if k == "bet":
            level = action[1]
            if name == gs.your_name:
                our = action[1]
        elif k == "raise":
            level = action[1]
            if name == gs.your_name:
                our = action[1]
        elif k == "call" and name == gs.your_name:
            our = level
    return our


def raise_to(gs, target):
    if gs.min_raise_to is None:
        return None
    our_w = our_street_wager(gs)
    max_to = our_w + gs.your_stack
    min_to = gs.min_raise_to
    t = int(max(min_to, min(target, max_to)))
    if t < min_to or t > max_to:
        return None
    return ("raise", t)


def clamp_bet(gs, amount):
    return max(1, min(int(amount), gs.your_stack))


def bet_amount(gs, fraction, floor=800):
    pot = max(gs.pot, 1)
    amt = int(pot * fraction)
    amt = max(amt, min(floor, gs.your_stack // 20))
    # Against empty pot, open to a fixed chip size
    if gs.pot == 0:
        amt = max(amt, floor)
    return clamp_bet(gs, amt)


def made_strength(hole, board):
    if len(board) < 3:
        return {
            "monster": 0.98,
            "premium": 0.90,
            "strong": 0.75,
            "playable": 0.48,
            "trash": 0.12,
        }[hand_category(hole)]

    score = evaluate_best(hole, board)
    cat = score[0]
    hole_ranks = {hole[0][1], hole[1][1]}
    board_ranks = [c[1] for c in board]
    top = max(board_ranks)

    hole_set = set(hole)
    uses_hole = False
    for combo in itertools.combinations(list(hole) + list(board), 5):
        if _evaluate_five(combo) == score and any(c in hole_set for c in combo):
            uses_hole = True
            break

    if cat >= 5:
        return 0.99 if uses_hole else 0.55
    if cat == 4:
        return 0.93 if uses_hole else 0.50
    if cat == 3:
        return 0.88 if uses_hole else 0.48
    if cat == 2:
        return 0.84 if uses_hole else 0.45
    if cat == 1:
        return 0.72 if uses_hole else 0.40
    if cat == 0:
        pair_rank = score[1]
        if pair_rank in hole_ranks and uses_hole:
            if hole[0][1] == hole[1][1]:
                # Pocket pair
                if pair_rank > top:
                    return 0.80
                if pair_rank == top:
                    return 0.70
                return 0.50 + pair_rank / 50.0
            if pair_rank == top:
                return 0.68
            if pair_rank >= 11:
                return 0.55
            return 0.42
        return 0.25
    # High card
    if max(hole_ranks) == 14:
        return 0.20
    return 0.08


def draw_score(hole, board):
    if not (3 <= len(board) <= 4):
        return 0.0
    cards = list(hole) + list(board)
    suits = Counter(c[0] for c in cards)
    flush_draw = max(suits.values()) == 4 and max(Counter(c[0] for c in hole).values()) >= 1

    ranks = sorted(set(c[1] for c in cards))
    if 14 in ranks:
        ranks = sorted(set(ranks + [1]))
    oesd = False
    for i, start in enumerate(ranks):
        window = [r for r in ranks if start <= r <= start + 4]
        if len(window) >= 4:
            oesd = True
            break
    pot = 0.0
    if flush_draw:
        pot += 0.38
    if oesd:
        pot += 0.28
    if flush_draw and oesd:
        pot = 0.72
    return min(pot, 0.75)


def estimate_equity(hole, board, n_opp, samples=80):
    if n_opp <= 0:
        return 1.0
    used = set(hole) | set(board)
    deck = [c for c in FULL_DECK if c not in used]
    need = 5 - len(board)
    wins = ties = trials = 0
    samples = max(30, min(samples, 100))

    for _ in range(samples):
        random.shuffle(deck)
        idx = 0
        opps = []
        if idx + 2 * n_opp + need > len(deck):
            continue
        for _o in range(n_opp):
            opps.append((deck[idx], deck[idx + 1]))
            idx += 2
        full = list(board) + deck[idx : idx + need]
        our = evaluate_best(hole, full)
        best_opp = max(evaluate_best(h, full) for h in opps)
        if our > best_opp:
            wins += 1
        elif our == best_opp:
            ties += 1
        trials += 1
    if trials == 0:
        return 0.5
    return (wins + 0.5 * ties) / trials


def last_aggressor_size(gs):
    """Most recent bet/raise size faced."""
    for name, action in reversed(gs.action_history):
        if name == gs.your_name:
            continue
        if action[0] == "bet":
            return action[1], name
        if action[0] == "raise":
            return action[1], name
    return gs.amount_to_call, None


def is_likely_bluff_range(gs, name):
    """True if this opponent fires huge with near-random range."""
    if name is None:
        return False
    st = _OPP[name]
    acts = st["folds"] + st["calls"] + st["bets"] + st["raises"]
    if acts < 4:
        # Prior matching big_bet / similar
        return True
    agg = (st["bets"] + st["raises"]) / max(acts, 1)
    avg = sum(st["sizes"]) / len(st["sizes"]) if st["sizes"] else 0
    return agg >= 0.35 and avg >= 2500


def decide_preflop(gs):
    hole = gs.your_hole_cards
    cat = hand_category(hole)
    to_call = gs.amount_to_call
    n_opp = len(active_opps(gs))
    station, maniac = table_profile(gs)
    chen = chen_score(hole)

    if to_call == 0:
        if cat == "monster":
            # Jam value vs stations; still large otherwise
            size = 12000 if station >= 0.55 else 7000
            return ("bet", clamp_bet(gs, size))
        if cat == "premium":
            return ("bet", clamp_bet(gs, 8000 if station >= 0.55 else 5000))
        if cat == "strong":
            return ("bet", clamp_bet(gs, 4500))
        if cat == "playable":
            # Take initiative before maniacs auto-bet
            return ("bet", clamp_bet(gs, 2200 if n_opp <= 4 else 1500))
        # Occasional steal with decent chen when first / few left
        if chen >= 4 and n_opp <= 2 and maniac >= 0.3:
            return ("bet", clamp_bet(gs, 1800))
        return ("check",)

    pot = max(gs.pot, 1)
    pot_odds = to_call / (pot + to_call)
    commit = to_call / max(gs.your_stack, 1)
    size, aggro = last_aggressor_size(gs)
    vs_maniac = is_likely_bluff_range(gs, aggro) or maniac >= 0.45

    if cat == "monster":
        our_w = our_street_wager(gs)
        if gs.min_raise_to is not None:
            # Stack off / huge raise vs stations & maniacs
            want = our_w + gs.your_stack if station >= 0.5 else our_w + to_call + max(pot * 2, 8000)
            action = raise_to(gs, want)
            if action:
                return action
        return ("call",)

    if cat == "premium":
        if gs.min_raise_to is not None and to_call <= 15000:
            our_w = our_street_wager(gs)
            action = raise_to(gs, our_w + to_call + max(pot, 5000))
            if action:
                return action
        if to_call <= 20000 or pot_odds <= 0.42:
            return ("call",)
        return ("fold",)

    if cat == "strong":
        # Call maniac opens very wide
        if vs_maniac and to_call <= 8000:
            return ("call",)
        if to_call <= 6000 or pot_odds <= 0.30:
            return ("call",)
        if station >= 0.6 and to_call <= 10000:
            return ("call",)
        return ("fold",)

    if cat == "playable":
        if vs_maniac and to_call <= 5500 and chen >= 5:
            return ("call",)
        if to_call <= 2000 or (to_call <= 4000 and pot_odds <= 0.25):
            return ("call",)
        return ("fold",)

    # Trash — only peel tiny bets or maniac min-prices with live cards
    if vs_maniac and to_call <= 5000 and chen >= 4.5:
        return ("call",)
    if to_call <= 300 and commit < 0.01:
        return ("call",)
    return ("fold",)


def decide_postflop(gs):
    hole = gs.your_hole_cards
    board = gs.community_cards
    n_opp = max(1, len(active_opps(gs)))
    station, maniac = table_profile(gs)
    strength = made_strength(hole, board)
    draw = draw_score(hole, board)

    samples = 60 if gs.street == "river" else (70 if gs.street == "turn" else 80)
    equity = estimate_equity(hole, board, min(n_opp, 3), samples=samples)

    # Blend
    raw = 0.5 * strength + 0.5 * equity + 0.25 * draw
    raw = min(0.99, raw)

    if gs.amount_to_call == 0:
        if strength >= 0.90 or equity >= 0.82:
            frac = 1.2 if station >= 0.5 else 0.9
            return ("bet", bet_amount(gs, frac, floor=5000))
        if strength >= 0.72 or equity >= 0.68:
            frac = 0.85 if station >= 0.5 else 0.65
            return ("bet", bet_amount(gs, frac, floor=3000))
        if strength >= 0.58 or equity >= 0.55:
            return ("bet", bet_amount(gs, 0.45, floor=1500))
        # Semi-bluff
        if draw >= 0.45 and n_opp <= 2 and gs.street != "river":
            return ("bet", bet_amount(gs, 0.55, floor=2000))
        # Stab weak boards heads-up when maniacs would auto-bet later
        if n_opp == 1 and equity >= 0.48 and maniac >= 0.35 and gs.street == "flop":
            return ("bet", bet_amount(gs, 0.35, floor=1500))
        return ("check",)

    # Facing a bet
    to_call = gs.amount_to_call
    pot = max(gs.pot, 1)
    pot_odds = to_call / (pot + to_call)
    commit = to_call / max(gs.your_stack, 1)
    size, aggro = last_aggressor_size(gs)
    vs_maniac = is_likely_bluff_range(gs, aggro) or (maniac >= 0.45 and size >= 3000)

    if strength >= 0.88 or equity >= 0.80:
        our_w = our_street_wager(gs)
        if gs.min_raise_to is not None:
            want = our_w + gs.your_stack if station >= 0.55 else our_w + to_call + max(int(pot * 1.2), to_call)
            action = raise_to(gs, want)
            if action:
                return action
        return ("call",)

    if strength >= 0.68 or equity >= 0.62:
        if vs_maniac and to_call <= 12000:
            if gs.min_raise_to is not None and strength >= 0.75:
                our_w = our_street_wager(gs)
                action = raise_to(gs, our_w + to_call + max(pot, 4000))
                if action:
                    return action
            return ("call",)
        if equity + 0.05 >= pot_odds or strength >= 0.70:
            return ("call",)
        if commit < 0.2:
            return ("call",)
        return ("fold",)

    # Draws / medium
    if draw >= 0.35 and pot_odds <= 0.35 and commit < 0.25:
        return ("call",)
    if vs_maniac and to_call <= 5500 and (equity >= 0.42 or strength >= 0.40 or chen_score(hole) >= 6):
        return ("call",)
    if equity >= pot_odds + 0.07 and commit < 0.3:
        return ("call",)
    if to_call <= max(500, pot * 0.1) and equity >= 0.38:
        return ("call",)
    return ("fold",)


def nextMove(gameState):
    try:
        if gameState.street == "preflop":
            return decide_preflop(gameState)
        return decide_postflop(gameState)
    except Exception:
        if gameState.amount_to_call == 0:
            return ("check",)
        return ("fold",)

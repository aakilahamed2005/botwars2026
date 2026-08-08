# Elite two-player Spades bot.
# Monte Carlo trick estimation + conservative contracts + bag/nil/score-aware play.

import random
from collections import defaultdict

SUITS = ("H", "D", "C", "S")
SPADES = "S"
FULL_DECK = [
    (s, r)
    for s in SUITS
    for r in range(2, 15)
    if not (s in ("C", "D") and r == 2)
]


def nextMove(gameState):
    if gameState.phase == "bid":
        return _bid(gameState)
    return _play(gameState)


# ---------------------------------------------------------------------------
# Bidding
# ---------------------------------------------------------------------------
def _bid(gs):
    hand = list(gs.your_hand)
    expected = _estimate_tricks(hand, samples=120)

    # Nil on weak, duckable hands.
    if expected <= 1.85 and _nil_safe(hand):
        bid = 0
    else:
        # Conservative: missing a bid costs 10*B; overtricks only give bags.
        raw = expected - 0.95
        bid = int(raw)  # floor
        if raw - bid >= 0.85:
            bid += 1
        bid = max(1, min(13, bid))

        # Tiny contracts are awkward — prefer nil when safe, else 2.
        if bid == 1 and expected < 2.4:
            bid = 0 if _nil_safe(hand) else 2 if expected >= 2.0 else 1

    # Second seat: shade against huge opponent bids / bag pressure.
    if gs.opponent_bid_known and gs.opponent_bid is not None:
        ob = gs.opponent_bid
        if ob == 0 and bid == 0:
            if expected >= 1.2:
                bid = max(1, min(3, int(expected - 0.5)))
        elif ob >= 9 and bid >= 4:
            bid = min(bid, max(2, 12 - ob))
        elif ob >= 7 and bid >= 5:
            bid = max(1, bid - 1)

    bags = gs.your_bags
    score = gs.your_score
    opp_score = gs.opponent_score

    if bid > 0 and bags >= 7:
        bid = min(bid, max(1, int(expected - 1.5)))
    if bid > 0 and score + bid * 10 >= 500 and score >= opp_score:
        bid = min(bid, max(1, int(expected - 1.2)))
    if bid > 0 and score <= -120 and expected >= 3.5:
        bid = max(bid, min(13, int(expected - 0.6)))

    if bid == 0 and not _nil_safe(hand):
        bid = max(1, min(3, int(expected - 1.0)))

    return max(0, min(13, bid))


def _nil_safe(hand):
    spades = [c for c in hand if c[0] == SPADES]
    if any(c[1] >= 13 for c in spades):
        return False
    if len(spades) >= 5:
        return False
    if sum(1 for c in spades if c[1] >= 11) >= 2:
        return False
    for s in ("H", "D", "C"):
        suit = [c for c in hand if c[0] == s]
        if not suit:
            continue
        if len(suit) == 1 and suit[0][1] >= 13:
            return False
        if len(suit) <= 2 and suit[0][1] == 14 and max(c[1] for c in suit) == 14:
            # singleton/doubleton Ace is a nil killer often
            if len(suit) == 1:
                return False
    highs = sum(1 for c in hand if c[1] >= 13)
    return highs <= 2


def _estimate_tricks(hand, samples=120):
    unknown = [c for c in FULL_DECK if c not in hand]
    total = 0.0
    seed = 0
    for c in hand:
        seed = (seed * 131 + hash(c)) & 0xFFFFFFFF
    rng = random.Random(seed ^ 0xC0FFEE)
    n = min(samples, 160)
    for _ in range(n):
        rng.shuffle(unknown)
        opp = unknown[:13]
        total += _sim_tricks(list(hand), list(opp), rng)
    return total / n


def _sim_tricks(me, opp, rng):
    my_tricks = 0
    spades_broken = False
    leader_is_me = rng.random() < 0.5
    for _ in range(13):
        if not me or not opp:
            break
        if leader_is_me:
            lead = _policy_lead(me, spades_broken)
            me.remove(lead)
            follow = _policy_follow(opp, lead, spades_broken)
            opp.remove(follow)
            if lead[0] == SPADES or follow[0] == SPADES:
                spades_broken = True
            if _lead_wins(lead, follow):
                my_tricks += 1
            else:
                leader_is_me = False
        else:
            lead = _policy_lead(opp, spades_broken)
            opp.remove(lead)
            follow = _policy_follow(me, lead, spades_broken)
            me.remove(follow)
            if lead[0] == SPADES or follow[0] == SPADES:
                spades_broken = True
            if not _lead_wins(lead, follow):
                my_tricks += 1
                leader_is_me = True
    return my_tricks


def _policy_lead(hand, spades_broken):
    legal = _legal(hand, None, spades_broken)
    non = [c for c in legal if c[0] != SPADES]
    if non:
        return max(non, key=lambda c: c[1])
    return max(legal, key=lambda c: c[1])


def _policy_follow(hand, lead, spades_broken):
    legal = _legal(hand, lead, spades_broken)
    winners = [c for c in legal if _beats(c, lead)]
    if winners:
        same = [c for c in winners if c[0] == lead[0]]
        if same:
            return min(same, key=lambda c: c[1])
        return min(winners, key=lambda c: c[1])
    return min(legal, key=lambda c: (c[0] == SPADES, c[1]))


def _lead_wins(lead, follow):
    if lead[0] == SPADES or follow[0] == SPADES:
        if lead[0] == SPADES and follow[0] == SPADES:
            return lead[1] >= follow[1]
        return lead[0] == SPADES
    if follow[0] == lead[0]:
        return lead[1] >= follow[1]
    return True


def _beats(card, lead):
    return not _lead_wins(lead, card)


# ---------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------
def _play(gs):
    hand = list(gs.your_hand)
    trick = list(gs.current_trick)
    lead_card = trick[0][1] if trick else None
    legal = _legal(hand, lead_card, gs.spades_broken)
    if not legal:
        return hand[0]

    info = _analyze(gs)
    goal = _goal(gs, info)

    if lead_card is None:
        return _choose_lead(legal, goal, gs, info)
    return _choose_follow(legal, lead_card, goal, gs, info)


def _analyze(gs):
    played = []
    for t in gs.trick_history:
        for _, card in t["plays"]:
            played.append(card)
    for _, card in gs.current_trick:
        played.append(card)

    remaining = set(FULL_DECK) - set(gs.your_hand) - set(played)
    tricks_done = len(gs.trick_history)
    opp_cards_left = 13 - tricks_done - (1 if gs.current_trick else 0)

    opp_voids = set()
    opp_name = gs.opponent_name
    for t in gs.trick_history:
        lead_suit = t["plays"][0][1][0]
        for name, card in t["plays"]:
            if name == opp_name and card[0] != lead_suit:
                opp_voids.add(lead_suit)

    rem_by_suit = defaultdict(list)
    for c in remaining:
        rem_by_suit[c[0]].append(c)

    return {
        "remaining": remaining,
        "rem_by_suit": rem_by_suit,
        "opp_voids": opp_voids,
        "opp_cards_left": max(0, opp_cards_left),
        "tricks_left": 13 - tricks_done,
    }


def _goal(gs, info):
    my_bid = gs.your_bid if gs.your_bid is not None else 0
    opp_bid = gs.opponent_bid if gs.opponent_bid is not None else 0
    my_tricks = gs.tricks_won.get(gs.your_name, 0)
    opp_tricks = gs.tricks_won.get(gs.opponent_name, 0)
    need = max(0, my_bid - my_tricks) if my_bid > 0 else 0
    tricks_left = info["tricks_left"]
    bags = gs.your_bags
    my_score = gs.your_score

    if my_bid == 0:
        return "nil"
    if opp_bid == 0 and opp_tricks == 0:
        if need > 0 and need >= tricks_left:
            return "must_win"
        # Still chase nil-bust if we can afford it.
        if need <= max(0, tricks_left - 2):
            return "bust_nil"
        return "win"
    if need > 0:
        if need >= tricks_left:
            return "must_win"
        if need >= tricks_left - 1:
            return "win"
        return "win"
    # Contract made — duck; duck harder near bag penalty or match clinch.
    over = my_tricks - my_bid
    if bags + over >= 7 or my_score + my_bid * 10 >= 470:
        return "duck_hard"
    return "duck"


def _choose_lead(legal, goal, gs, info):
    if goal in ("nil", "duck", "duck_hard"):
        return _exit_lead(legal, info, hard=(goal != "duck"))

    if goal == "bust_nil":
        # Lead into known voids to force a trump/dump, else lowest soft lead.
        for s in list(info["opp_voids"]):
            if s == SPADES:
                continue
            mine = [c for c in legal if c[0] == s]
            if mine:
                return min(mine, key=lambda c: c[1])
        return _safe_dump(legal, preserve_spades=True)

    if goal == "must_win":
        sure = _cashable(legal, info, gs)
        if sure:
            return sure[0]
        if gs.spades_broken:
            sp = [c for c in legal if c[0] == SPADES]
            if sp:
                return max(sp, key=lambda c: c[1])
        non = [c for c in legal if c[0] != SPADES]
        if non:
            return max(non, key=lambda c: c[1])
        return max(legal, key=lambda c: c[1])

    return _best_offensive_lead(legal, gs, info)


def _choose_follow(legal, lead, goal, gs, info):
    winners = [c for c in legal if _beats(c, lead)]
    losers = [c for c in legal if not _beats(c, lead)]

    if goal == "nil":
        if losers:
            same = [c for c in losers if c[0] == lead[0]]
            if same:
                return max(same, key=lambda c: c[1])
            return _safe_dump(losers, preserve_spades=True)
        return min(winners, key=lambda c: (c[0] == SPADES, c[1]))

    if goal in ("duck", "duck_hard"):
        if losers:
            same = [c for c in losers if c[0] == lead[0]]
            if same:
                # duck: waste medium; duck_hard: keep exits, play lowest
                return (
                    min(same, key=lambda c: c[1])
                    if goal == "duck_hard"
                    else max(same, key=lambda c: c[1])
                )
            return _safe_dump(losers, preserve_spades=True)
        return _cheapest_winner(winners, lead)

    if goal == "bust_nil":
        if winners:
            return _cheapest_winner(winners, lead)
        return _safe_dump(legal, preserve_spades=False)

    # win / must_win
    if winners:
        if goal == "must_win":
            same = [c for c in winners if c[0] == lead[0]]
            if same:
                return max(same, key=lambda c: c[1])
            return max(winners, key=lambda c: c[1])
        return _cheapest_winner(winners, lead)
    return _safe_dump(legal, preserve_spades=True)


def _cheapest_winner(winners, lead):
    same = [c for c in winners if c[0] == lead[0]]
    if same:
        return min(same, key=lambda c: c[1])
    return min(winners, key=lambda c: c[1])


def _safe_dump(cards, preserve_spades=True):
    if preserve_spades:
        return sorted(cards, key=lambda c: (c[0] == SPADES, c[1], c[0]))[0]
    return sorted(cards, key=lambda c: (c[1], c[0] == SPADES))[0]


def _exit_lead(legal, info, hard=False):
    """Lead a card least likely to win (for nil / bag avoidance)."""
    # Prefer long weak side suits; avoid leading spades; avoid lone high cards.
    non = [c for c in legal if c[0] != SPADES]
    pool = non if non else legal
    by = defaultdict(list)
    for c in pool:
        by[c[0]].append(c)

    def suit_key(s):
        cards = by[s]
        return (max(c[1] for c in cards), -len(cards), min(c[1] for c in cards))

    # Lead lowest card from the weakest-looking suit.
    suit = min(by.keys(), key=suit_key)
    cards = by[suit]
    return min(cards, key=lambda c: c[1])


def _cashable(legal, info, gs):
    out = []
    for c in legal:
        suit, rank = c
        higher = [x for x in info["rem_by_suit"].get(suit, []) if x[1] > rank]
        if suit == SPADES and not higher:
            out.append(c)
        elif not higher and rank == 14 and suit not in info["opp_voids"]:
            out.append(c)
    out.sort(key=lambda c: (c[0] != SPADES, -c[1]))
    return out


def _best_offensive_lead(legal, gs, info):
    for s in ("H", "D", "C"):
        mine = [c for c in legal if c[0] == s]
        if not mine:
            continue
        top = max(mine, key=lambda c: c[1])
        higher_out = [c for c in info["rem_by_suit"].get(s, []) if c[1] > top[1]]
        if top[1] >= 14 and not higher_out and s not in info["opp_voids"]:
            return top
        if top[1] >= 13 and not higher_out and len(mine) >= 2:
            return top

    if gs.spades_broken:
        sp = [c for c in legal if c[0] == SPADES]
        if sp:
            # Pull trump with a mid/high spade when we still need tricks.
            return max(sp, key=lambda c: c[1])

    non = [c for c in legal if c[0] != SPADES]
    if non:
        by = defaultdict(list)
        for c in non:
            by[c[0]].append(c)
        suit = max(by.keys(), key=lambda s: (len(by[s]), max(x[1] for x in by[s])))
        return max(by[suit], key=lambda c: c[1])
    return max(legal, key=lambda c: c[1])


def _legal(hand, lead_card, spades_broken):
    if lead_card is None:
        non = [c for c in hand if c[0] != SPADES]
        if not non or spades_broken:
            return list(hand)
        return non
    same = [c for c in hand if c[0] == lead_card[0]]
    return same if same else list(hand)

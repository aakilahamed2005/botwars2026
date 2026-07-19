import random


# ---------------------------------------------------------------------------
# Card constants / helpers (mirrors engine.py exactly so legality/scoring
# calculations match the real engine bit for bit)
# ---------------------------------------------------------------------------
SUITS = ["H", "D", "C", "S"]
RANKS = list(range(2, 15))  # 2..14, 14 = Ace
FULL_DECK = [(s, r) for s in SUITS for r in RANKS]


def resolve_trick(lead_card, follow_card, trump_suit):
    """True if lead_card wins, False if follow_card wins. (Copied from engine.py)"""
    lead_suit, lead_rank = lead_card
    follow_suit, follow_rank = follow_card

    if follow_suit == lead_suit:
        return lead_rank >= follow_rank
    elif follow_suit == trump_suit:
        return False
    else:
        return True


def legal_cards(hand, lead_card):
    """Cards a player may legally play. (Copied from engine.py)"""
    if lead_card is None:
        return list(hand)
    lead_suit = lead_card[0]
    same_suit = [c for c in hand if c[0] == lead_suit]
    return same_suit if same_suit else list(hand)


def card_strength(card, trump_suit):
    """A single comparable 'value' for a card — trumps always outrank
    every non-trump card, and within a category higher rank is better."""
    suit, rank = card
    return rank + (100 if suit == trump_suit else 0)


# ---------------------------------------------------------------------------
# Persistent memory across calls within one game
# ---------------------------------------------------------------------------
_MEM = {}


def _get_memory(view):
    """Return this game's memory dict, resetting it if a brand-new game
    has just started (detected from the known state at the very first
    trick: phase 1, 25 cards left in the (post-face-up-reveal) stock,
    a full 13-card hand, and no scoring tricks recorded yet)."""
    global _MEM

    is_fresh_game = (
        view.phase == 1
        and view.stock_remaining == 25
        and len(view.your_hand) == 13
        and view.tricks_won.get(view.your_name, 0) == 0
        and view.tricks_won.get(view.opponent_name, 0) == 0
    )

    if is_fresh_game or "my_played" not in _MEM:
        _MEM = {
            "my_played": set(),
            "opp_known_played": set(),
        }
    return _MEM


def _record_opponent_lead(view, mem):
    """If the current trick already has a card in it, it must have been
    led by the opponent (the engine only calls us when it's our turn, and
    a non-empty trick at that point always means the other player led).
    That's the only time we're ever actually shown one of the opponent's
    cards, so we squirrel it away."""
    if view.current_trick:
        leader_name, leader_card = view.current_trick[0]
        if leader_name == view.opponent_name:
            mem["opp_known_played"].add(leader_card)


def unseen_cards(view, mem):
    """Every card that is not provably accounted for: this is the pool the
    opponent's hand (plus, in phase 1, the hidden stock) is drawn from."""
    known = set(view.your_hand)
    known |= mem["my_played"]
    known |= mem["opp_known_played"]
    if view.face_up_card:
        known.add(view.face_up_card)
    return [c for c in FULL_DECK if c not in known]


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------
def opponent_can_beat_lead(lead_card, opp_hand, trump_suit):
    """Given a hypothetical opponent hand, could they legally beat our
    lead card?"""
    suit, rank = lead_card
    same_suit = [c for c in opp_hand if c[0] == suit]
    if same_suit:
        return any(c[1] > rank for c in same_suit)
    if suit == trump_suit:
        return False  # nothing beats a trump lead once you're void in it
    return any(c[0] == trump_suit for c in opp_hand)


def win_prob_as_leader(card, trump_suit, unseen_pool, opp_hand_size, n_samples=150):
    """Monte-Carlo estimate of P(this lead card wins the trick)."""
    if opp_hand_size <= 0 or not unseen_pool:
        return 1.0
    k = min(opp_hand_size, len(unseen_pool))
    if k <= 0:
        return 1.0
    wins = 0
    for _ in range(n_samples):
        sample = random.sample(unseen_pool, k)
        if not opponent_can_beat_lead(card, sample, trump_suit):
            wins += 1
    return wins / n_samples


def face_up_value(card, trump_suit):
    """How good the currently-contested face-up card is."""
    if card is None:
        return -1.0
    suit, rank = card
    return rank + (50 if suit == trump_suit else 0)


def attack_threshold(stock_remaining):
    """Get more willing to fight for a mediocre card as the stock runs
    low — there are fewer future chances to draw something better."""
    progress = max(0.0, min(1.0, (25 - stock_remaining) / 25.0))
    return 9.0 - 3.0 * progress


def choose_dump_card(cards, trump_suit):
    """Cheapest, least useful card available — used both for deliberate
    recruitment-phase losses and for plain discards."""
    return min(cards, key=lambda c: card_strength(c, trump_suit))


# ---------------------------------------------------------------------------
# Greedy policy used inside Monte-Carlo rollouts (stands in for "reasonable
# play" by both sides so a rollout can be carried out to completion quickly)
# ---------------------------------------------------------------------------
def _greedy_lead(hand, trump_suit):
    return choose_dump_card(hand, trump_suit)


def _greedy_follow(allowed, lead_card, trump_suit):
    winners = [c for c in allowed if resolve_trick(lead_card, c, trump_suit) is False]
    if winners:
        return min(winners, key=lambda c: card_strength(c, trump_suit))
    return choose_dump_card(allowed, trump_suit)


def simulate_phase2_playout(my_hand, opp_hand, trump_suit, leader_is_me):
    """Roll the rest of the scoring phase forward with the greedy policy on
    both sides. Returns (my_extra_tricks, opp_extra_tricks)."""
    my_hand = list(my_hand)
    opp_hand = list(opp_hand)
    my_tricks = 0
    opp_tricks = 0

    while my_hand and opp_hand:
        if leader_is_me:
            lead_card = _greedy_lead(my_hand, trump_suit)
            my_hand.remove(lead_card)
            allowed = legal_cards(opp_hand, lead_card)
            follow_card = _greedy_follow(allowed, lead_card, trump_suit)
            opp_hand.remove(follow_card)
            lead_wins = resolve_trick(lead_card, follow_card, trump_suit)
            winner_is_me = lead_wins
        else:
            lead_card = _greedy_lead(opp_hand, trump_suit)
            opp_hand.remove(lead_card)
            allowed = legal_cards(my_hand, lead_card)
            follow_card = _greedy_follow(allowed, lead_card, trump_suit)
            my_hand.remove(follow_card)
            lead_wins = resolve_trick(lead_card, follow_card, trump_suit)
            winner_is_me = not lead_wins

        if winner_is_me:
            my_tricks += 1
        else:
            opp_tricks += 1
        leader_is_me = winner_is_me

    return my_tricks, opp_tricks


# ---------------------------------------------------------------------------
# Decision functions
# ---------------------------------------------------------------------------
def choose_lead_card_phase1(view, mem):
    hand = view.your_hand
    if len(hand) == 1:
        return hand[0]

    trump = view.trump_suit
    threshold = attack_threshold(view.stock_remaining)

    if face_up_value(view.face_up_card, trump) >= threshold:
        unseen = unseen_cards(view, mem)
        opp_size = min(len(hand), len(unseen))
        best_card, best_score = None, -1.0
        for c in sorted(hand, key=lambda c: card_strength(c, trump)):
            p = win_prob_as_leader(c, trump, unseen, opp_size, n_samples=150)
            score = p - 0.0005 * card_strength(c, trump)
            if score > best_score:
                best_score, best_card = score, c
        return best_card if best_card is not None else choose_dump_card(hand, trump)

    return choose_dump_card(hand, trump)


def choose_lead_card_phase2(view, mem):
    hand = list(view.your_hand)
    if len(hand) == 1:
        return hand[0]

    trump = view.trump_suit
    unseen = unseen_cards(view, mem)
    opp_size = min(len(hand), len(unseen))

    n_samples = max(10, min(40, 400 // max(1, len(hand))))

    best_card, best_avg = None, -1.0
    for card in hand:
        remaining_my_hand = [c for c in hand if c != card]
        total, trials = 0.0, 0
        for _ in range(n_samples):
            k = min(opp_size, len(unseen))
            if k <= 0:
                break
            opp_sample = random.sample(unseen, k)
            allowed = legal_cards(opp_sample, card)
            if not allowed:
                continue
            follow_card = _greedy_follow(allowed, card, trump)
            lead_wins = resolve_trick(card, follow_card, trump)
            opp_after = [c for c in opp_sample if c != follow_card]

            my_tricks = 1 if lead_wins else 0
            rest_me, _ = simulate_phase2_playout(remaining_my_hand, opp_after, trump, lead_wins)
            my_tricks += rest_me

            total += my_tricks
            trials += 1

        avg = (total / trials) if trials else 0.0
        if avg > best_avg:
            best_avg, best_card = avg, card

    return best_card if best_card is not None else choose_dump_card(hand, trump)


def choose_follow_card(view, mem, lead_card):
    hand = view.your_hand
    trump = view.trump_suit
    allowed = legal_cards(hand, lead_card)
    if len(allowed) == 1:
        return allowed[0]

    winners = [c for c in allowed if resolve_trick(lead_card, c, trump) is False]

    if view.phase == 2:
        if winners:
            return min(winners, key=lambda c: card_strength(c, trump))
        return choose_dump_card(allowed, trump)

    # Phase 1 (recruitment)
    threshold = attack_threshold(view.stock_remaining)
    if winners and face_up_value(view.face_up_card, trump) >= threshold:
        return min(winners, key=lambda c: card_strength(c, trump))
    return choose_dump_card(allowed, trump)


# ---------------------------------------------------------------------------
# Entry point required by the engine
# ---------------------------------------------------------------------------
def nextMove(gameState):
    view = gameState
    try:
        mem = _get_memory(view)
        _record_opponent_lead(view, mem)

        if view.current_trick:
            lead_card = view.current_trick[-1][1]
            card = choose_follow_card(view, mem, lead_card)
        else:
            if view.phase == 1:
                card = choose_lead_card_phase1(view, mem)
            else:
                card = choose_lead_card_phase2(view, mem)

        lead_card_for_check = view.current_trick[-1][1] if view.current_trick else None
        allowed = legal_cards(view.your_hand, lead_card_for_check)
        if card not in allowed:
            card = allowed[0]

        mem["my_played"].add(card)
        return card

    except Exception:
        # Never forfeit due to a bug — always fall back to a safe legal move.
        lead_card_for_check = view.current_trick[-1][1] if view.current_trick else None
        allowed = legal_cards(view.your_hand, lead_card_for_check)
        return allowed[0]
# Stronger two-player Spades bot with contract-aware bidding and trick control.


def nextMove(gameState):
    if gameState.phase == "bid":
        return _bid_move(gameState)
    return _play_move(gameState)


def _bid_move(gameState):
    hand = gameState.your_hand
    if not hand:
        return 0

    suit_cards = {suit: [card for card in hand if card[0] == suit] for suit in ("H", "D", "C", "S")}
    spades = suit_cards["S"]
    off_suits = [suit_cards[s] for s in ("H", "D", "C")]

    score = 0.0

    # Trump strength.
    for card in sorted(spades, key=lambda c: c[1], reverse=True):
        rank = card[1]
        if rank >= 14:
            score += 1.8
        elif rank >= 13:
            score += 1.5
        elif rank >= 12:
            score += 1.2
        elif rank >= 11:
            score += 0.9
        elif rank >= 10:
            score += 0.6
        elif rank >= 8:
            score += 0.25

    if len(spades) >= 5:
        score += 0.9
    elif len(spades) >= 4:
        score += 0.55
    elif len(spades) == 3:
        score += 0.2
    elif len(spades) <= 1:
        score -= 0.4

    # Side suit strength.
    for suit in off_suits:
        if not suit:
            continue
        high = max(card[1] for card in suit)
        if high >= 14:
            score += 0.85
        elif high >= 13:
            score += 0.7
        elif high >= 12:
            score += 0.45
        elif high >= 10:
            score += 0.2

        if len(suit) >= 4:
            score += 0.35
        elif len(suit) == 3:
            score += 0.2
        elif len(suit) == 2:
            score += 0.08

        if len(suit) <= 2 and any(card[1] >= 12 for card in suit):
            score += 0.15

    # Hand shape / control.
    high_cards = sum(1 for card in hand if card[1] >= 12)
    if high_cards >= 4:
        score += 0.8
    elif high_cards >= 2:
        score += 0.3

    if len(spades) <= 2 and high_cards <= 1:
        score -= 1.0

    if score <= 1.4 and high_cards <= 1 and len(spades) <= 2:
        return 0

    bid = int(round(score))
    bid = max(0, min(13, bid))

    if gameState.opponent_bid_known and gameState.opponent_bid is not None:
        if gameState.opponent_bid >= 10 and bid >= 4:
            bid = min(bid, 3)
        elif gameState.opponent_bid >= 8 and bid >= 5:
            bid = max(0, bid - 1)

    if bid == 1 and score < 1.7:
        return 0
    if bid == 2 and score < 2.2 and high_cards <= 1:
        return 0

    return bid


def _play_move(gameState):
    hand = list(gameState.your_hand)
    trick = list(gameState.current_trick)
    legal_cards = _legal_cards(hand, trick, gameState.spades_broken)

    my_bid = gameState.your_bid if gameState.your_bid is not None else 0
    opp_bid = gameState.opponent_bid if gameState.opponent_bid is not None else 0
    my_tricks = gameState.tricks_won.get(gameState.your_name, 0)

    if my_bid == 0:
        goal = "lose"
    elif opp_bid == 0:
        goal = "bust_nil"
    elif my_tricks < my_bid:
        goal = "win"
    else:
        goal = "lose"

    if not trick:
        return _play_lead(legal_cards, goal, gameState)
    return _play_follow(legal_cards, trick[0][1], goal, gameState)


def _legal_cards(hand, trick, spades_broken):
    if not trick:
        non_spades = [card for card in hand if card[0] != "S"]
        if not non_spades or spades_broken:
            return list(hand)
        return non_spades

    lead_suit = trick[0][1][0]
    same_suit = [card for card in hand if card[0] == lead_suit]
    if same_suit:
        return same_suit
    return list(hand)


def _play_lead(legal_cards, goal, gameState):
    if not legal_cards:
        return None

    if goal in {"lose", "bust_nil"}:
        return _lowest_safe_card(legal_cards)

    # If we need to take the trick, lead the highest card we can afford.
    if gameState.spades_broken:
        if any(card[0] == "S" for card in legal_cards):
            spades = [card for card in legal_cards if card[0] == "S"]
            return max(spades, key=lambda c: c[1])
        return max(legal_cards, key=lambda c: c[1])

    # Before spades are broken, pull with a strong off-suit if available.
    non_spades = [card for card in legal_cards if card[0] != "S"]
    if non_spades:
        return max(non_spades, key=lambda c: c[1])
    return max(legal_cards, key=lambda c: c[1])


def _play_follow(legal_cards, lead_card, goal, gameState):
    if not legal_cards:
        return None

    lead_suit = lead_card[0]
    lead_rank = lead_card[1]

    if goal in {"lose", "bust_nil"}:
        same_suit_losing = [card for card in legal_cards if card[0] == lead_suit and card[1] < lead_rank]
        if same_suit_losing:
            return max(same_suit_losing, key=lambda c: c[1])
        return _lowest_safe_card(legal_cards)

    winning_cards = []
    for card in legal_cards:
        if card[0] == lead_suit and card[1] > lead_rank:
            winning_cards.append(card)
        elif card[0] == "S" and lead_suit != "S":
            winning_cards.append(card)

    if winning_cards:
        same_suit_winners = [card for card in winning_cards if card[0] == lead_suit]
        if same_suit_winners:
            return min(same_suit_winners, key=lambda c: c[1])
        return min(winning_cards, key=lambda c: (c[1], c[0] == "S"))

    return _lowest_safe_card(legal_cards)


def _lowest_safe_card(legal_cards):
    # Prefer dumping a low off-suit over burning a low spade.
    ordered = sorted(legal_cards, key=lambda card: (card[0] == "S", card[1], card[0]))
    return ordered[0]

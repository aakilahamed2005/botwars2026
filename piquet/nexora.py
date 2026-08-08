# point_maximizer.py
def nextMove(view):
    hand = view.your_hand
    
    if view.phase == "exchange":
        # Determine max discard allowed
        if view.turn == view.elder:
            max_d = min(5, len(hand))
        else:
            max_d = min(view.talon_remaining, len(hand))
            
        # Group by suit to find the longest
        suits = {"H": [], "D": [], "C": [], "S": []}
        for c in hand:
            suits[c[0]].append(c)
            
        longest_suit = max(suits.keys(), key=lambda s: len(suits[s]))
        
        # Identify discard candidates: anything not in the longest suit, lowest ranks first
        candidates = [c for c in hand if c[0] != longest_suit]
        candidates.sort(key=lambda c: c[1]) 
        
        return candidates[:max_d]

    elif view.phase == "declare":
        # Safety check: Engine forfeits invalid claims. Only claim if we actually hold the combination.
        cat = view.declare_category
        if cat == "point":
            return "claim" # Always have at least 1 card
        elif cat == "sequence":
            has_seq = False
            suits = {"H": [], "D": [], "C": [], "S": []}
            for c in hand: suits[c[0]].append(c[1])
            for s, ranks in suits.items():
                ranks = sorted(list(set(ranks)))
                count = 1
                for i in range(1, len(ranks)):
                    if ranks[i] == ranks[i-1] + 1:
                        count += 1
                        if count >= 3: has_seq = True
                    else:
                        count = 1
            return "claim" if has_seq else "pass"
        elif cat == "set":
            counts = {}
            has_set = False
            for c in hand:
                if c[1] >= 10:
                    counts[c[1]] = counts.get(c[1], 0) + 1
                    if counts[c[1]] >= 3: has_set = True
            return "claim" if has_set else "pass"

    elif view.phase == "tricks":
        if not view.current_trick: # Leading
            # Lead the highest card of our longest suit
            suits = {"H": [], "D": [], "C": [], "S": []}
            for c in hand: suits[c[0]].append(c)
            longest = max(suits.keys(), key=lambda s: len(suits[s]))
            return max(suits[longest], key=lambda c: c[1])
        else: # Following
            lead_suit = view.current_trick[0][1][0]
            lead_rank = view.current_trick[0][1][1]
            legal = [c for c in hand if c[0] == lead_suit]
            if not legal:
                legal = hand # Void in lead suit
                
            # If we can follow suit and win, play the lowest winning card
            if legal[0][0] == lead_suit:
                winners = [c for c in legal if c[1] > lead_rank]
                if winners:
                    return min(winners, key=lambda c: c[1])
            
            # Otherwise, dump the lowest legal card
            return min(legal, key=lambda c: c[1])
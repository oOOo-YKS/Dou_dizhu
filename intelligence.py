import random
from itertools import combinations

class Intelligence:
    @staticmethod
    def _counts(cards):
        d = {}
        for c in cards:
            d[c] = d.get(c, 0) + 1
        return d

    @staticmethod
    def _is_consecutive_no_high(ranks):
        if not ranks or max(ranks) >= 12:
            return False
        return all(ranks[i] + 1 == ranks[i + 1] for i in range(len(ranks) - 1))

    @staticmethod
    def _consecutive_segments(sorted_ranks):
        if not sorted_ranks:
            return []
        segs = [[sorted_ranks[0]]]
        for r in sorted_ranks[1:]:
            if r == segs[-1][-1] + 1:
                segs[-1].append(r)
            else:
                segs.append([r])
        return segs

    @staticmethod
    def _judge_category(cards):
        if cards is None or len(cards) == 0:
            return {"type": "pass", "length": 0}

        cards = sorted(list(cards))
        cnt = Intelligence._counts(cards)
        uniq = sorted(cnt.keys())
        counts_desc = sorted(cnt.values(), reverse=True)
        n = len(cards)

        # Rocket
        if set(cards) == {13, 14}:
            return {"type": "rocket", "length": 2}

        # Bomb
        if n == 4 and counts_desc == [4]:
            r = uniq[0]
            return {"type": "bomb", "length": 4, "main": [r]}

        # Single / Pair / Triple
        if n == 1:
            return {"type": "single", "length": 1, "main": [cards[0]]}
        if n == 2 and counts_desc == [2]:
            return {"type": "pair", "length": 2, "main": [uniq[0]]}
        if n == 3 and counts_desc == [3]:
            return {"type": "triple", "length": 3, "main": [uniq[0]]}

        # Triple + Single / Pair
        if n == 4 and counts_desc == [3, 1]:
            main = [k for k, v in cnt.items() if v == 3][0]
            return {"type": "triple_one", "length": 4, "main": [main]}
        if n == 5 and counts_desc == [3, 2]:
            main = [k for k, v in cnt.items() if v == 3][0]
            return {"type": "triple_pair", "length": 5, "main": [main]}

        # Straight
        if n >= 5 and all(v == 1 for v in cnt.values()):
            if Intelligence._is_consecutive_no_high(uniq):
                return {"type": "straight", "length": n, "main": uniq[:]}

        # Double straight
        if n >= 6 and n % 2 == 0 and all(v == 2 for v in cnt.values()):
            if Intelligence._is_consecutive_no_high(uniq):
                return {"type": "double_straight", "length": n, "main": uniq[:]}

        # Airplanes
        trip_ok_ranks = [r for r in range(0, 12) if cnt.get(r, 0) >= 3]
        if trip_ok_ranks:
            segments = Intelligence._consecutive_segments(trip_ok_ranks)
            for seg in segments:
                if len(seg) < 2:
                    continue
                for i in range(len(seg)):
                    for j in range(i + 2, len(seg) + 1):
                        main_trip = seg[i:j]
                        k = len(main_trip)
                        rem = cnt.copy()
                        for r in main_trip:
                            rem[r] -= 3
                            if rem[r] == 0:
                                del rem[r]
                        rem_total = sum(rem.values())
                        if rem_total == 0 and n == 3 * k:
                            return {"type": "airplane", "length": n, "main": main_trip[:]}
                        if rem_total == k and all(v == 1 for v in rem.values()) and not any(r in main_trip for r in rem.keys()):
                            if n == 4 * k:
                                return {"type": "airplane_solo_wings", "length": n, "main": main_trip[:], "wings": sorted(rem.keys())}
                        if rem_total == 2 * k and all(v == 2 for v in rem.values()) and not any(r in main_trip for r in rem.keys()):
                            if n == 5 * k:
                                return {"type": "airplane_pair_wings", "length": n, "main": main_trip[:], "wings": sorted(rem.keys())}

        return {"type": "unknown", "length": n}

    def give_risk_state(self, cards: list) -> int:
        """For landlord bidding: simple heuristic (based on high cards)."""
        high_cards = sum(1 for c in cards if c >= 10)  # 10, J, Q, K, A, 2, jokers
        if high_cards > 8:
            return 3
        elif high_cards > 5:
            return 2
        else:
            return 1

    def play_first(self, cards: list, judge_fn):
        """Start a new round: pick a random valid move, prefer winning if possible."""
        possible = self._generate_all_moves(cards, judge_fn=judge_fn)
        if not possible:
            return None
        # Prefer moves that empty hand
        winning_moves = [m for m in possible if len(cards) - len(m) == 0]
        if winning_moves:
            return random.choice(winning_moves)
        return random.choice(possible)

    def respond(self, cards: list, last_play_cat, judge_fn):
        """Respond to last play: pick a valid beating move, prefer winning if possible."""
        possible = self._generate_all_moves(cards, last_play=last_play_cat, judge_fn=judge_fn)
        if not possible:
            return None
        # Prefer moves that empty hand
        winning_moves = [m for m in possible if len(cards) - len(m) == 0]
        if winning_moves:
            return random.choice(winning_moves)
        return random.choice(possible)

    def _generate_all_moves(self, cards: list, last_play=None, judge_fn=None):
        """Generate all valid moves given current cards and last play category."""
        moves = []
        cnt = self._counts(cards)

        # --- Singles, pairs, triples, bombs ---
        for u, c in cnt.items():
            if c >= 1:
                moves.append([u])
            if c >= 2:
                moves.append([u, u])
            if c >= 3:
                moves.append([u, u, u])
            if c >= 4:
                moves.append([u] * 4)

        # --- Rocket ---
        if 13 in cnt and 14 in cnt:
            moves.append([13, 14])

        # --- Triple + single ---
        triple_ranks = [r for r in cnt if cnt[r] >= 3]
        single_ranks = [r for r in cnt if cnt[r] >= 1]
        for u in triple_ranks:
            for s in [r for r in single_ranks if r != u]:
                moves.append([u] * 3 + [s])

        # --- Triple + pair ---
        pair_ranks = [r for r in cnt if cnt[r] >= 2]
        for u in triple_ranks:
            for p in [r for r in pair_ranks if r != u]:
                moves.append([u] * 3 + [p] * 2)

        # --- Singles straights (5+ consecutive, no 2/jokers) ---
        single_ranks = sorted([r for r in range(0, 12) if cnt.get(r, 0) >= 1])
        for length in range(5, len(single_ranks) + 1):
            for i in range(len(single_ranks) - length + 1):
                seq = single_ranks[i:i + length]
                if all(seq[j] + 1 == seq[j + 1] for j in range(len(seq) - 1)):
                    moves.append(seq)

        # --- Double straights (3+ consecutive pairs, no 2/jokers) ---
        pair_ranks = sorted([r for r in range(0, 12) if cnt.get(r, 0) >= 2])
        for length in range(3, len(pair_ranks) + 1):
            for i in range(len(pair_ranks) - length + 1):
                seq = pair_ranks[i:i + length]
                if all(seq[j] + 1 == seq[j + 1] for j in range(len(seq) - 1)):
                    moves.append([r for r in seq for _ in range(2)])

        # --- Airplanes (2+ consecutive triples, optional wings) ---
        trip_ranks = sorted([r for r in range(0, 12) if cnt.get(r, 0) >= 3])
        segments = self._consecutive_segments(trip_ranks)
        for seg in segments:
            if len(seg) < 2:
                continue
            for start in range(len(seg)):
                for end in range(start + 2, len(seg) + 1):
                    main_trip = seg[start:end]
                    k = len(main_trip)
                    # Pure airplane
                    airplane_cards = [r for r in main_trip for _ in range(3)]
                    moves.append(airplane_cards)
                    # Remaining ranks not in main triples
                    avail_solo_ranks = [r for r in cnt if r not in main_trip and cnt[r] >= 1]
                    avail_pair_ranks = [r for r in cnt if r not in main_trip and cnt[r] >= 2]

                    # --- Solo wings ---
                    if len(avail_solo_ranks) >= k:
                        for wings in combinations(avail_solo_ranks, k):
                            wing_cards = list(wings)
                            moves.append(airplane_cards + wing_cards)

                    # --- Pair wings ---
                    if len(avail_pair_ranks) >= k:
                        for wings in combinations(avail_pair_ranks, k):
                            wing_cards = [r for r in wings for _ in range(2)]
                            moves.append(airplane_cards + wing_cards)

        # --- Filter by last_play if exists ---
        if last_play is not None:
            filtered = []
            last_type = last_play["type"]
            last_len = last_play["length"]
            last_main = last_play.get("main", [])
            for mv in moves:
                cat = judge_fn(mv)
                if cat["type"] == "rocket":
                    filtered.append(mv)
                    continue
                if cat["type"] == "bomb" and last_type not in ["bomb", "rocket"]:
                    filtered.append(mv)
                    continue
                if cat["type"] == last_type and cat["length"] == last_len:
                    if max(cat.get("main", [0])) > max(last_main):
                        filtered.append(mv)
            moves = filtered

        return moves
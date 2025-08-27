import random
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from datetime import datetime

class PlayerInfo:
    def __init__(self, my_cards: list, moves: list):
        self.my_cards = my_cards
        self.moves = moves

    def last_play(self):
        for rel_id, cards in reversed(self.moves):
            if cards is not None:
                return rel_id, cards
        return None

class Intelligence:
    @staticmethod
    def _counts(cards):
        """Count occurrences of each card rank."""
        d = {}
        for c in cards:
            d[c] = d.get(c, 0) + 1
        return d

    @staticmethod
    def _is_consecutive_no_high(ranks):
        """Check if ranks are consecutive and below 2 (rank 12)."""
        if not ranks or max(ranks) >= 12:
            return False
        return len(ranks) == ranks[-1] - ranks[0] + 1

    @staticmethod
    def _consecutive_segments(sorted_ranks):
        """Group consecutive ranks into segments."""
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
        """Determine the category of a played card combination."""
        if cards is None or not cards:
            return {"type": "pass", "length": 0}

        cards = sorted(cards)
        cnt = Intelligence._counts(cards)
        uniq = sorted(cnt.keys())
        counts_desc = sorted(cnt.values(), reverse=True)
        n = len(cards)

        if set(cards) == {13, 14}:
            return {"type": "rocket", "length": 2}

        if n == 4 and counts_desc == [4]:
            return {"type": "bomb", "length": 4, "main": [uniq[0]]}

        if n == 1:
            return {"type": "single", "length": 1, "main": [cards[0]]}
        if n == 2 and counts_desc == [2]:
            return {"type": "pair", "length": 2, "main": [uniq[0]]}
        if n == 3 and counts_desc == [3]:
            return {"type": "triple", "length": 3, "main": [uniq[0]]}

        if n == 4 and counts_desc == [3, 1]:
            main = next(k for k, v in cnt.items() if v == 3)
            return {"type": "triple_one", "length": 4, "main": [main]}
        if n == 5 and counts_desc == [3, 2]:
            main = next(k for k, v in cnt.items() if v == 3)
            return {"type": "triple_pair", "length": 5, "main": [main]}

        if n >= 5 and all(v == 1 for v in cnt.values()):
            if Intelligence._is_consecutive_no_high(uniq):
                return {"type": "straight", "length": n, "main": uniq}

        if n >= 6 and n % 2 == 0 and all(v == 2 for v in cnt.values()):
            if Intelligence._is_consecutive_no_high(uniq):
                return {"type": "double_straight", "length": n, "main": uniq}

        trip_ok_ranks = {r for r in range(12) if cnt.get(r, 0) >= 3}
        if trip_ok_ranks:
            segments = Intelligence._consecutive_segments(sorted(trip_ok_ranks))
            for seg in segments:
                if len(seg) < 2:
                    continue
                for i in range(len(seg) - 1):
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
                            return {"type": "airplane", "length": n, "main": main_trip}
                        if rem_total == k and all(v == 1 for v in rem.values()) and not any(r in main_trip for r in rem):
                            if n == 4 * k:
                                return {"type": "airplane_solo_wings", "length": n, "main": main_trip, "wings": sorted(rem.keys())}
                        if rem_total == 2 * k and all(v == 2 for v in rem.values()) and not any(r in main_trip for r in rem):
                            if n == 5 * k:
                                return {"type": "airplane_pair_wings", "length": n, "main": main_trip, "wings": sorted(rem.keys())}

        return {"type": "unknown", "length": n}

    def give_risk_state(self, pinfo: PlayerInfo) -> int:
        """For landlord bidding: heuristic based on high cards."""
        high_cards = sum(1 for c in pinfo.my_cards if c >= 10)
        return 3 if high_cards > 8 else 2 if high_cards > 5 else 1

    def play_first(self, pinfo: PlayerInfo, judge_fn):
        """Start a new round: pick a random valid move, prefer winning."""
        possible = Intelligence._generate_all_moves(pinfo.my_cards, judge_fn=judge_fn)
        if not possible:
            return None
        winning_moves = [m for m in possible if len(pinfo.my_cards) == len(m)]
        return random.choice(winning_moves or possible)

    def respond(self, pinfo: PlayerInfo, last_play_cat, judge_fn):
        """Respond to last play: pick a valid beating move, prefer winning."""
        possible = Intelligence._generate_all_moves(pinfo.my_cards, last_play=last_play_cat, judge_fn=judge_fn)
        if not possible:
            return None
        winning_moves = [m for m in possible if len(pinfo.my_cards) == len(m)]
        return random.choice(winning_moves or possible)

    @staticmethod
    def _generate_all_moves(cards: list, last_play=None, judge_fn=None):
        """Generate all valid moves given current cards and last play category."""
        moves = []
        cnt = Intelligence._counts(cards)
        ranks = set(cnt)

        for u, c in cnt.items():
            if c >= 1:
                moves.append([u])
            if c >= 2:
                moves.append([u, u])
            if c >= 3:
                moves.append([u] * 3)
            if c >= 4:
                moves.append([u] * 4)

        if 13 in ranks and 14 in ranks:
            moves.append([13, 14])

        triple_ranks = {r for r in cnt if cnt[r] >= 3}
        single_ranks = ranks
        for u in triple_ranks:
            for s in single_ranks - {u}:
                moves.append([u] * 3 + [s])

        pair_ranks = {r for r in cnt if cnt[r] >= 2}
        for u in triple_ranks:
            for p in pair_ranks - {u}:
                moves.append([u] * 3 + [p] * 2)

        single_ranks = sorted(r for r in range(12) if r in cnt)
        for length in range(5, len(single_ranks) + 1):
            for i in range(len(single_ranks) - length + 1):
                seq = single_ranks[i:i + length]
                if len(seq) == seq[-1] - seq[0] + 1:
                    moves.append(seq)

        pair_ranks = sorted(r for r in range(12) if cnt.get(r, 0) >= 2)
        for length in range(3, len(pair_ranks) + 1):
            for i in range(len(pair_ranks) - length + 1):
                seq = pair_ranks[i:i + length]
                if len(seq) == seq[-1] - seq[0] + 1:
                    moves.append([r for r in seq for _ in range(2)])

        trip_ranks = sorted(r for r in range(12) if cnt.get(r, 0) >= 3)
        segments = Intelligence._consecutive_segments(trip_ranks)
        for seg in segments:
            if len(seg) < 2:
                continue
            for start in range(len(seg) - 1):
                for end in range(start + 2, len(seg) + 1):
                    main_trip = seg[start:end]
                    k = len(main_trip)
                    airplane_cards = [r for r in main_trip for _ in range(3)]
                    moves.append(airplane_cards)
                    avail_solo_ranks = [r for r in cnt if r not in main_trip and cnt[r] >= 1]
                    avail_pair_ranks = [r for r in cnt if r not in main_trip and cnt[r] >= 2]
                    if len(avail_solo_ranks) >= k:
                        for wings in combinations(avail_solo_ranks, k):
                            moves.append(airplane_cards + list(wings))
                    if len(avail_pair_ranks) >= k:
                        for wings in combinations(avail_pair_ranks, k):
                            wing_cards = [r for r in wings for _ in range(2)]
                            moves.append(airplane_cards + wing_cards)

        if last_play:
            filtered = []
            filtered.append([])
            last_type, last_len, last_main = last_play["type"], last_play["length"], last_play.get("main", [])
            for mv in moves:
                cat = judge_fn(mv)
                if cat["type"] == "rocket":
                    filtered.append(mv)
                elif cat["type"] == "bomb" and last_type not in ["bomb", "rocket"]:
                    filtered.append(mv)
                elif cat["type"] == last_type and cat["length"] == last_len:
                    if max(cat.get("main", [0])) > max(last_main):
                        filtered.append(mv)
            moves = filtered

        return moves

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(195, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        y1 = self.fc1(x)
        r1 = F.relu(y1)
        d1 = self.dropout(r1)
        y2 = self.fc2(d1)
        r2 = F.relu(y2)
        return r2


class HistoryRnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=18, hidden_size=128, num_layers=2, batch_first=False)
        self.fc = nn.Linear(128, 80)

    def forward(self, x):
        # x: (seq_len, 18)
        out, _ = self.rnn(x.unsqueeze(1))  # Add "batch" as 1 for GRU (shape: (seq_len, 1, 18))
        last_out = out[-1, 0, :]  # Get last timestep output: shape (128,)
        embedding = self.fc(last_out)  # shape (80,)
        return embedding
    
class HandModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(15, 30)

    def forward(self, x):
        # x: (15,)
        return self.fc(x)  # shape (30,)
    
class StateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = HistoryRnnModel()
        self.hand = HandModel()
        self.fc = nn.Linear(80 + 30, 128)  # Combine history and hand

    def forward(self, history_enc, hand_enc):
        # history_enc: (seq_len, 18)
        # hand_enc: (15,)
        history_info = self.history(history_enc)  # (80,)
        hand_info = self.hand(hand_enc)  # (30,)
        combined = torch.cat([history_info, hand_info], dim=-1)  # (110,)
        return self.fc(combined)  # (128,)

        

class BidHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        return self.fc(x)

class PolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128 + 15 + 1, 1)

    def forward(self, x):
        return self.fc(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.grade = 0
        self.state = StateModel()
        self.bid_head = BidHead()
        self.play_first_head = PolicyHead()
        self.respond_head = PolicyHead()
        self.train_mode = False

    def save(self, folder_path):
        """Save the model state to a file."""
        os.makedirs(folder_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(folder_path, f"model_{timestamp}.pth")
        torch.save(self.state_dict(), file_path)
        return file_path

    def load(self, file_path):
        """Load the model state from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model file found at {file_path}")
        self.load_state_dict(torch.load(file_path))
        return self

    def variation(self):
        """Create a perturbed copy of the model."""
        new_model = copy.deepcopy(self)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.01)
                    param.requires_grad_(False)
        return new_model

    def encode_hand(self, cards):
        """Encode cards as a 15-dim vector of counts."""
        count = torch.zeros(15, dtype=torch.float)
        for c in cards:
            count[c] += 1
        return count

    def encode_action(self, cards):
        """Encode a move as a 15-dim vector of counts."""
        return self.encode_hand(cards)

    def encode_lord(self, lord: int):
        return torch.tensor([lord + 1], dtype=torch.float)

    def encode_state(self, pinfo: PlayerInfo):
        """Encode player state (hand + move history)."""
        hand_enc = self.encode_hand(pinfo.my_cards)
        moves = pinfo.moves
        if len(moves) == 0:
            history_enc = torch.zeros(1, 18)  # shape (1, 18)
        else:
            history_enc = torch.zeros(len(moves), 18, dtype=torch.float)
        for i, (rel_id, mcards) in enumerate(moves):
            move_enc = torch.zeros(15) if mcards is None else self.encode_hand(mcards)
            rel_onehot = torch.zeros(3)
            rel_onehot[rel_id + 1] = 1
            history_enc[i] = torch.cat([rel_onehot, move_enc])
        
        return self.state(history_enc, hand_enc)

    def give_risk_state(self, pinfo: PlayerInfo) -> int:
        """Predict bidding risk (1, 2, 3)."""
        state = self.encode_state(pinfo)
        logits = self.bid_head(state)
        return (torch.multinomial(F.softmax(logits, dim=-1), 1).item() + 1) if self.train_mode else torch.argmax(logits).item() + 1

    def play_first(self, pinfo: PlayerInfo, lord, judge_fn):
        """Select a move to start a round."""
        possible = Intelligence._generate_all_moves(pinfo.my_cards, judge_fn=judge_fn)
        if not possible:
            return None
        state = self.encode_state(pinfo)
        lord_enc = self.encode_lord(lord)
        scores = torch.stack([self.play_first_head(torch.cat([state, self.encode_action(mv), lord_enc])) for mv in possible])
        idx = torch.multinomial(F.softmax(scores.squeeze(-1), dim=0), 1).item() if self.train_mode else torch.argmax(scores).item()
        return possible[idx]

    def respond(self, pinfo: PlayerInfo, lord, last_play_cat, judge_fn):
        """Select a move to beat last_play_cat."""
        possible = Intelligence._generate_all_moves(pinfo.my_cards, last_play=last_play_cat, judge_fn=judge_fn)
        if not possible:
            return None
        state = self.encode_state(pinfo)
        lord_enc = self.encode_lord(lord)
        scores = torch.stack([self.respond_head(torch.cat([state, self.encode_action(mv), lord_enc])) for mv in possible])
        idx = torch.multinomial(F.softmax(scores.squeeze(-1), dim=0), 1).item() if self.train_mode else torch.argmax(scores).item()
        return possible[idx]
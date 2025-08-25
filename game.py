import random
import torch
from intelligence import Intelligence, Model, PlayerInfo

CARD_NAMES = {
    0: '3', 1: '4', 2: '5', 3: '6', 4: '7', 5: '8', 6: '9', 7: '10',
    8: 'J', 9: 'Q', 10: 'K', 11: 'A', 12: '2', 13: 'joker', 14: 'JOKER'
}

class Deck:
    def __init__(self):
        self.cards = [i for i in range(13) for _ in range(4)] + [13, 14]
        random.shuffle(self.cards)

class GameInfo:
    def __init__(self):
        self.risk = [[0, 0] for _ in range(3)]
        self.moves = []

    def add_move(self, player_id: int, played_cards):
        self.moves.append((player_id, played_cards))

    def to_player_info(self, player_id: int, my_cards: list):
        rel_moves = []
        for pid, cards in self.moves:
            rel_id = 0 if pid == player_id else 1 if (pid - player_id) % 3 == 1 else -1
            rel_moves.append((rel_id, cards))
        return PlayerInfo(my_cards=my_cards, moves=rel_moves)

class Player:
    def __init__(self, index: int, cards: list, intelligence):
        self.index = index
        self.cards = cards
        self.intelligence = intelligence

    def remove_cards(self, played):
        for c in played:
            self.cards.remove(c)

    def has_cards(self):
        return len(self.cards) > 0

class Game:
    def __init__(self, p1_intelligence, p2_intelligence, p3_intelligence, show_process=False):
        self.show_process = show_process
        self.end = False
        self.info = GameInfo()
        deck = Deck()
        self.players = [
            Player(0, deck.cards[0:17], p1_intelligence),
            Player(1, deck.cards[17:34], p2_intelligence),
            Player(2, deck.cards[34:51], p3_intelligence)
        ]
        self.reserved_cards = deck.cards[51:]
        self.starting_player = random.choice([0, 1, 2])
        if self.show_process:
            for p in self.players:
                print(f"Player {p.index} init cards: {self.show_cards(p.cards)}")
            print(f"Reserved cards: {self.show_cards(self.reserved_cards)}\n")

        for i in range(3):
            player = self.players[(self.starting_player + i) % 3]
            pinfo = self.info.to_player_info(player.index, player.cards)
            risk = player.intelligence.give_risk_state(pinfo)
            if self.show_process:
                print(f"Player{(self.starting_player + i) % 3} bid {risk}")
            self.info.risk[i] = (self.starting_player + i) % 3, risk

        self.lord = self._choose_a_lord()
        if self.show_process:
            print(f"Player{self.lord} is the lord")
        self.players[self.lord].cards.extend(self.reserved_cards)
        self.playing_role = self.lord
        self.last_play = None
        self.passing_count = 0
        for i, r in enumerate(self.info.risk):
            if r[0] == self.lord:
                self.risk = r[1]
                break

    def _choose_a_lord(self):
        for priority in [3, 2, 1]:
            for i, r in enumerate(self.info.risk):
                if r[1] == priority:
                    return r[0]
        return self.starting_player

    def turn(self):
        current_player = self.players[self.playing_role]
        if not current_player.has_cards():
            return

        judge_fn = Intelligence._judge_category
        pinfo = self.info.to_player_info(self.playing_role, current_player.cards)
        lord = 0 if self.playing_role == self.lord else 1 if (self.lord - self.playing_role) % 3 == 1 else -1

        if self.last_play is None or self.passing_count == 2:
            chosen = current_player.intelligence.play_first(pinfo, judge_fn=judge_fn, lord=lord)
            if chosen:
                self.last_play = judge_fn(chosen)
                self.last_play["player"] = self.playing_role
                if self.last_play["type"] in ["rocket", "bomb"]:
                    self.risk *= 2
                current_player.remove_cards(chosen)
                if self.show_process:
                    print(f"Player {self.playing_role} plays {self.show_cards(chosen)}")
                self.info.add_move(self.playing_role, chosen)
                self.passing_count = 0
            else:
                if self.show_process:
                    print(f"Player {self.playing_role} passes")
                self.info.add_move(self.playing_role, None)
                self.passing_count += 1
        else:
            chosen = current_player.intelligence.respond(pinfo, lord=lord, last_play_cat=self.last_play, judge_fn=judge_fn)
            if chosen:
                self.last_play = judge_fn(chosen)
                self.last_play["player"] = self.playing_role
                if self.last_play["type"] in ["rocket", "bomb"]:
                    self.risk *= 2
                current_player.remove_cards(chosen)
                if self.show_process:
                    print(f"Player {self.playing_role} beats with {self.show_cards(chosen)}")
                self.info.add_move(self.playing_role, chosen)
                self.passing_count = 0
            else:
                if self.show_process:
                    print(f"Player {self.playing_role} passes")
                self.info.add_move(self.playing_role, None)
                self.passing_count += 1

        if not current_player.has_cards():
            self.end = True
            is_lord = self.playing_role == self.lord
            lord_player = self.players[self.lord]
            next_player = self.players[(self.lord + 1) % 3]
            prev_player = self.players[(self.lord - 1) % 3]
            if is_lord:
                lord_player.intelligence.grade += self.risk * 2
                next_player.intelligence.grade -= self.risk
                prev_player.intelligence.grade -= self.risk
            else:
                lord_player.intelligence.grade -= self.risk * 2
                next_player.intelligence.grade += self.risk
                prev_player.intelligence.grade += self.risk
            winner = "Landlord" if is_lord else "Peasants"
            print(f"Game Over! {winner} win! (Player {self.playing_role} emptied hand)")

        self.playing_role = (self.playing_role + 1) % 3

    def show_cards(self, cards):
        return [CARD_NAMES[c] for c in cards]

    def play_game(self):
        while not self.end:
            self.turn()

if __name__ == "__main__":
    i1 = Model()
    i2 = Model()
    i3 = Model()
    game = Game(i1, i2, i3, show_process=True)
    print("Initial Hands:")
    for p in game.players:
        print(f"Player {p.index}: {game.show_cards(p.cards)}")
    print(f"Reserved Cards: {game.show_cards(game.reserved_cards)}")
    print(f"Landlord: Player {game.lord}\n")
    game.play_game()
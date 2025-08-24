import numpy as np
import numpy.random as npr
from intelligence import Intelligence

CARD_NAMES = {
    0:'3', 1:'4', 2:'5', 3:'6', 4:'7', 5:'8', 6:'9', 7:'10', 8:'J', 9:'Q',
    10:'K', 11:'A', 12:'2', 13:'joker', 14:'JOKER'
}

class Deck:
    def __init__(self):
        self.cards = np.array([
            *([i for i in range(13) for _ in range(4)]), # 0-12 repeated 4 times
            13, 14 # jokers
        ])
        npr.shuffle(self.cards)

class GameInfo:
    def __init__(self):
        self.risk = np.zeros((3, 2), dtype=int)
        self.moves = []  # history of moves

    def add_move(self, player_id: int, played_cards):
        """played_cards is list[int] or None if pass"""
        self.moves.append((player_id, played_cards))

    def to_player_info(self, player_id: int, my_cards: np.ndarray):
        """Transform to relative perspective"""
        rel_moves = []
        for pid, cards in self.moves:
            # Convert to relative index
            if pid == player_id:
                rel_id = 0
            elif (pid - player_id) % 3 == 1:
                rel_id = 1  # next player
            else:
                rel_id = -1  # last player
            rel_moves.append((rel_id, cards))

        return PlayerInfo(my_cards=my_cards, moves=rel_moves)


class PlayerInfo:
    def __init__(self, my_cards: np.ndarray, moves: list):
        """
        my_cards: current hand
        moves: [(relative_id, played_cards), ...]
        """
        self.my_cards = my_cards
        self.moves = moves

    def last_play(self):
        # Find last non-pass move
        for rel_id, cards in reversed(self.moves):
            if cards is not None:
                return rel_id, cards
        return None


class Player:
    def __init__(self, index: int, cards: np.array, intelligence: Intelligence):
        self.index = index
        self.cards = np.sort(cards)
        self.intelligence = intelligence

    def remove_cards(self, played):
        for c in played:
            self.cards = self.cards[self.cards != c]

    def has_cards(self):
        return len(self.cards) > 0

class Game:
    def __init__(self):
        self.end = False
        self.info = GameInfo()
        deck = Deck()
        self.p1 = Player(index=0, cards=deck.cards[0:17], intelligence=Intelligence())
        self.p2 = Player(index=1, cards=deck.cards[17:34], intelligence=Intelligence())
        self.p3 = Player(index=2, cards=deck.cards[34:51], intelligence=Intelligence())
        self.reserved_cards = np.sort(deck.cards[51:])
        self.starting_player = npr.choice(3)

        # Evaluate risk and choose landlord
        for i in range(3):
            player = [self.p1, self.p2, self.p3][(self.starting_player + i) % 3]
            risk = player.intelligence.give_risk_state(player.cards)
            self.info.risk[i, 0] = (self.starting_player + i) % 3
            self.info.risk[i, 1] = risk

        self.lord = self._choose_a_lord()
        [self.p1, self.p2, self.p3][self.lord].cards = np.sort(
            np.append([self.p1, self.p2, self.p3][self.lord].cards, self.reserved_cards)
        )
        self.playing_role = self.lord
        self.last_play = None  # (cards, type, player)
        self.passing_count = 0

    def _choose_a_lord(self):
        for priority in [3, 2, 1]:
            for i in range(3):
                if self.info.risk[i, 1] == priority:
                    return self.info.risk[i, 0]
        return self.starting_player

    def turn(self):
        current_player = [self.p1, self.p2, self.p3][self.playing_role]

        if self.last_play is None or self.passing_count == 2:
            # Start a new round
            chosen = current_player.intelligence.play_first(current_player.cards)
            if chosen:
                self.last_play = (chosen, self.playing_role)
                current_player.remove_cards(chosen)
                print(f"Player {self.playing_role} plays {self.show_cards(chosen)}")
            self.passing_count = 0
        else:
            # Respond to last play
            chosen = current_player.intelligence.respond(current_player.cards, self.last_play)
            if chosen:
                self.last_play = (chosen, self.playing_role)
                current_player.remove_cards(chosen)
                print(f"Player {self.playing_role} beats with {self.show_cards(chosen)}")
                self.passing_count = 0
            else:
                print(f"Player {self.playing_role} passes")
                self.passing_count += 1

        if not current_player.has_cards():
            print(f"Player {self.playing_role} wins!")
            self.end = True

        self.playing_role = (self.playing_role + 1) % 3

    def show_cards(self, cards):
        return [CARD_NAMES[c] for c in cards]

    def play_game(self):
        while not self.end:
            self.turn()

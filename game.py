import random
from intelligence import Intelligence

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
        self.risk = [[0, 0] for _ in range(3)]  # [player_id, risk]
        self.moves = []  # history: (player_id, cards or None)

    def add_move(self, player_id: int, played_cards):
        self.moves.append((player_id, played_cards))

    def to_player_info(self, player_id: int, my_cards: list):
        rel_moves = []
        for pid, cards in self.moves:
            if pid == player_id:
                rel_id = 0
            elif (pid - player_id) % 3 == 1:
                rel_id = 1
            else:
                rel_id = -1
            rel_moves.append((rel_id, cards))
        return PlayerInfo(my_cards=my_cards, moves=rel_moves)

class PlayerInfo:
    def __init__(self, my_cards: list, moves: list):
        self.my_cards = my_cards
        self.moves = moves

    def last_play(self):
        for rel_id, cards in reversed(self.moves):
            if cards is not None:
                return rel_id, cards
        return None

class Player:
    def __init__(self, index: int, cards: list, intelligence: Intelligence):
        self.index = index
        self.cards = sorted(cards)
        self.intelligence = intelligence

    def remove_cards(self, played):
        for c in played:
            self.cards.remove(c)
        self.cards.sort()

    def has_cards(self):
        return len(self.cards) > 0

class Game:
    def __init__(self):
        self.end = False
        self.info = GameInfo()
        deck = Deck()
        self.p1 = Player(0, deck.cards[0:17], Intelligence())
        self.p2 = Player(1, deck.cards[17:34], Intelligence())
        self.p3 = Player(2, deck.cards[34:51], Intelligence())
        self.reserved_cards = sorted(deck.cards[51:])
        self.starting_player = random.choice([0, 1, 2])

        # Risk evaluation for landlord
        for i in range(3):
            player = [self.p1, self.p2, self.p3][(self.starting_player + i) % 3]
            risk = player.intelligence.give_risk_state(player.cards)
            self.info.risk[i][0] = (self.starting_player + i) % 3
            self.info.risk[i][1] = risk

        self.lord = self._choose_a_lord()
        lord_player = [self.p1, self.p2, self.p3][self.lord]
        lord_player.cards = sorted(lord_player.cards + self.reserved_cards)
        self.playing_role = self.lord
        self.last_play = None
        self.passing_count = 0

    def _choose_a_lord(self):
        for priority in [3, 2, 1]:
            for i in range(3):
                if self.info.risk[i][1] == priority:
                    return self.info.risk[i][0]
        return self.starting_player

    def turn(self):
        current_player = [self.p1, self.p2, self.p3][self.playing_role]
        if not current_player.has_cards():
            return  # Skip if already won (though game should end before)

        judge_fn = Intelligence._judge_category  # Use shared judge
        if self.last_play is None or self.passing_count == 2:
            # New round
            chosen = current_player.intelligence.play_first(current_player.cards, judge_fn=judge_fn)
            if chosen:
                self.last_play = judge_fn(chosen)
                self.last_play["player"] = self.playing_role
                current_player.remove_cards(chosen)
                print(f"Player {self.playing_role} plays {self.show_cards(chosen)}")
                self.info.add_move(self.playing_role, chosen)
                self.passing_count = 0
            else:
                print(f"Player {self.playing_role} passes")
                self.info.add_move(self.playing_role, None)
                self.passing_count += 1
        else:
            # Respond
            chosen = current_player.intelligence.respond(current_player.cards, self.last_play, judge_fn=judge_fn)
            if chosen:
                self.last_play = judge_fn(chosen)
                self.last_play["player"] = self.playing_role
                current_player.remove_cards(chosen)
                print(f"Player {self.playing_role} beats with {self.show_cards(chosen)}")
                self.info.add_move(self.playing_role, chosen)
                self.passing_count = 0
            else:
                print(f"Player {self.playing_role} passes")
                self.info.add_move(self.playing_role, None)
                self.passing_count += 1

        # Check win after move
        if not current_player.has_cards():
            self.end = True
            winner = "Landlord" if self.playing_role == self.lord else "Peasants"
            print(f"\nGame Over! {winner} win! (Player {self.playing_role} emptied hand)")

        # Always advance to next player
        self.playing_role = (self.playing_role + 1) % 3

    def show_cards(self, cards):
        return [CARD_NAMES[c] for c in sorted(cards)]

    def play_game(self):
        while not self.end:
            self.turn()


if __name__ == "__main__":
    game = Game()
    print("Initial Hands:")
    print("Player 0:", game.show_cards(game.p1.cards))
    print("Player 1:", game.show_cards(game.p2.cards))
    print("Player 2:", game.show_cards(game.p3.cards))
    print(f"Reserved Cards: {game.show_cards(game.reserved_cards)}")
    print(f"Landlord: Player {game.lord}\n")
    game.play_game()
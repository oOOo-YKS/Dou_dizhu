from game import Game

def test_judge_category():
    game = Game()

    # --- Single ---
    assert game._judge_category([3])['type'] == 'single'
    assert game._judge_category([12])['type'] == 'single'  # '2'

    # --- Pair ---
    assert game._judge_category([3, 3])['type'] == 'pair'
    assert game._judge_category([8, 8])['type'] == 'pair'

    # --- Triple ---
    assert game._judge_category([4, 4, 4])['type'] == 'triple'

    # --- Bomb ---
    assert game._judge_category([5, 5, 5, 5])['type'] == 'bomb'

    # --- Triple + One ---
    assert game._judge_category([6, 6, 6, 7])['type'] == 'triple_one'

    # --- Triple + Pair ---
    assert game._judge_category([8, 8, 8, 9, 9])['type'] == 'triple_pair'

    # --- Straight ---
    assert game._judge_category([3, 4, 5, 6, 7])['type'] == 'straight'

    # --- Double Straight ---
    assert game._judge_category([3, 3, 4, 4, 5, 5])['type'] == 'double_straight'

    # --- Airplane (triple straight, no wings) ---
    # Example: 3-4-5 triple straight: 3,3,3,4,4,4,5,5,5
    airplane_cards = [3,3,3,4,4,4,5,5,5]
    assert game._judge_category(airplane_cards)['type'] == 'airplane'

    # --- Airplane + single wings ---
    # 3-4-5 triple + 3 single wings: 3,3,3,4,4,4,5,5,5,6,7,8
    airplane_solo_wings = [3,3,3,4,4,4,5,5,5,6,7,8]
    assert game._judge_category(airplane_solo_wings)['type'] == 'airplane_solo_wings'

    # --- Airplane + pair wings ---
    # 3-4-5 triple + 3 pair wings: 3,3,3,4,4,4,5,5,5,6,6,7,7,8,8
    airplane_pair_wings = [3,3,3,4,4,4,5,5,5,6,6,7,7,8,8]
    assert game._judge_category(airplane_pair_wings)['type'] == 'airplane_pair_wings'

    # --- Rocket ---
    assert game._judge_category([13, 14])['type'] == 'rocket'

    print("All test cases passed!")

if __name__ == "__main__":
    test_judge_category()
    hands = {
        "single": [3],
        "pair": [3, 3],
        "triple": [4, 4, 4],
        "bomb": [5, 5, 5, 5],
        "triple_one": [6, 6, 6, 7],
        "triple_pair": [8, 8, 8, 9, 9],
        "straight": [3, 4, 5, 6, 7],
        "double_straight": [3, 3, 4, 4, 5, 5],
        "airplane": [3,3,3,4,4,4,5,5,5],
        "airplane_solo_wings": [3,3,3,4,4,4,5,5,5,6,7,8],
        "airplane_pair_wings": [3,3,3,4,4,4,5,5,5,6,6,7,7,8,8],
        "rocket": [13, 14],
    }

    game = Game()
    for name, cards in hands.items():
        category = game._judge_category(cards)
        print(f"{name}: {cards} -> {category}")
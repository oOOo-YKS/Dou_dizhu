import torch
import torch.optim as optim
import torch.nn.functional as F
from game import Game
from intelligence import ModelIntelligence, Intelligence, PlayerInfo

def train_model(num_epochs=10, games_per_epoch=100, lr=0.001):
    """Train the model using REINFORCE by simulating games."""
    model = ModelIntelligence()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for g in range(games_per_epoch):
            model.train_mode = True
            game = Game(intelligence_class=lambda: model)
            log_probs = []
            player_roles = []
            actions = []

            # Wrap methods to collect log probabilities
            original_play_first = model.play_first
            original_respond = model.respond

            def wrapped_play_first(pinfo, judge_fn):
                if not isinstance(pinfo, PlayerInfo):
                    print(f"Error: pinfo is not a PlayerInfo instance: {type(pinfo)}")
                    return None
                if not isinstance(pinfo.my_cards, list):
                    print(f"Error: Invalid pinfo.my_cards: {pinfo.my_cards}")
                    return None
                print(f"Debug: Calling _generate_all_moves in play_first with cards={pinfo.my_cards}, judge_fn={judge_fn}")
                try:
                    possible = Intelligence._generate_all_moves(pinfo.my_cards, judge_fn=judge_fn)
                except TypeError as e:
                    print(f"TypeError in _generate_all_moves (play_first): {e}")
                    print(f"Cards passed: {pinfo.my_cards}")
                    return None
                if not possible:
                    return None
                state = model.encode_state(pinfo)
                feat = model.base(state)
                scores = []
                for mv in possible:
                    act_enc = model.encode_action(mv)
                    input_policy = torch.cat([feat, act_enc])
                    score = model.play_first_head(input_policy)
                    scores.append(score)
                scores = torch.stack(scores)
                probs = F.softmax(scores.squeeze(-1), dim=0)
                idx = torch.multinomial(probs, 1).item()
                log_probs.append(torch.log(probs[idx]))
                player_roles.append(game.playing_role)
                actions.append((True, possible[idx]))
                return possible[idx]

            def wrapped_respond(pinfo, last_play_cat, judge_fn):
                if not isinstance(pinfo, PlayerInfo):
                    print(f"Error: pinfo is not a PlayerInfo instance: {type(pinfo)}")
                    return None
                if not isinstance(pinfo.my_cards, list):
                    print(f"Error: Invalid pinfo.my_cards: {pinfo.my_cards}")
                    return None
                print(f"Debug: Calling _generate_all_moves in respond with cards={pinfo.my_cards}, last_play={last_play_cat}, judge_fn={judge_fn}")
                try:
                    possible = Intelligence._generate_all_moves(pinfo.my_cards, last_play=last_play_cat, judge_fn=judge_fn)
                except TypeError as e:
                    print(f"TypeError in _generate_all_moves (respond): {e}")
                    print(f"Cards passed: {pinfo.my_cards}")
                    return None
                if not possible:
                    return None
                state = model.encode_state(pinfo)
                feat = model.base(state)
                scores = []
                for mv in possible:
                    act_enc = model.encode_action(mv)
                    input_policy = torch.cat([feat, act_enc])
                    score = model.respond_head(input_policy)
                    scores.append(score)
                scores = torch.stack(scores)
                probs = F.softmax(scores.squeeze(-1), dim=0)
                idx = torch.multinomial(probs, 1).item()
                log_probs.append(torch.log(probs[idx]))
                player_roles.append(game.playing_role)
                actions.append((False, possible[idx]))
                return possible[idx]

            model.play_first = wrapped_play_first
            model.respond = wrapped_respond
            try:
                game.play_game()
            except Exception as e:
                print(f"Game simulation failed: {e}")
                continue
            model.play_first = original_play_first
            model.respond = original_respond

            # Assign rewards
            winner_player = (game.playing_role - 1) % 3  # Player who ended the game
            is_lord_win = winner_player == game.lord
            rewards = []
            for pr in player_roles:
                if pr == game.lord:
                    rewards.append(1 if is_lord_win else -1)
                else:
                    rewards.append(1 if not is_lord_win else -1)

            # Compute loss
            loss = 0
            for lp, r in zip(log_probs, rewards):
                loss += -lp * r
            loss = loss / len(log_probs) if log_probs else torch.tensor(0.0, requires_grad=True)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss {epoch_loss / games_per_epoch:.4f}")
    model.train_mode = False
    try:
        torch.save(model.state_dict(), "d:/Projects/Dou_dizhu/model.pt")
    except Exception as e:
        print(f"Failed to save model: {e}")
    return model

if __name__ == "__main__":
    # Test _generate_all_moves independently
    test_cards = [0, 0, 1, 2, 3, 4, 5]
    print(f"Testing _generate_all_moves with cards={test_cards}")
    try:
        moves = Intelligence._generate_all_moves(test_cards, judge_fn=Intelligence._judge_category)
        print(f"Possible moves: {moves}")
    except TypeError as e:
        print(f"Test failed: {e}")

    model = train_model(num_epochs=5, games_per_epoch=50)
    game = Game(intelligence_class=lambda: model)
    print("Initial Hands:")
    print("Player 0:", game.show_cards(game.p1.cards))
    print("Player 1:", game.show_cards(game.p2.cards))
    print("Player 2:", game.show_cards(game.p3.cards))
    print(f"Reserved Cards: {game.show_cards(game.reserved_cards)}")
    print(f"Landlord: Player {game.lord}\n")
    game.play_game()
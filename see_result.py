from intelligence import Model

models = [Model() for _ in range(3)]
for model in models:
    model.load(r"D:\Models\model_20250824_221140.pth")

from game import Game

g = Game(models[0], models[1], models[2], show_process=True)

g.play_game()
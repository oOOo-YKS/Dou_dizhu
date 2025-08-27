from intelligence import Model
import os
import glob


def get_most_recent_model(path="D:/Models"):
    try:
        # Find all .pth files in the directory
        model_files = glob.glob(os.path.join(path, "*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No .pth files found in {path}")
        
        # Get the file with the latest creation time
        most_recent_file = max(model_files, key=os.path.getctime)
        return most_recent_file
    except Exception as e:
        raise Exception(f"Error finding most recent model file in {path}: {str(e)}")
    

path = get_most_recent_model("D:/Models")
print(path)
models = [Model() for _ in range(3)]
for model in models:
    model.load(path)

from game import Game

g = Game(models[0], models[1], models[2], show_process=True)

g.play_game()
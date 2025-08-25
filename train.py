import os
import glob
from intelligence import Model
from game import Game

def get_most_recent_model(path="D:/Models"):
    """Return the path to the most recent .pth file in the specified directory."""
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

def loop():
    """Play games with all unique model triplets and update models."""
    for i in range(model_num - 2):
        for j in range(i + 1, model_num - 1):
            for h in range(j + 1, model_num):
                print(f"Playing game with models {i}, {j}, {h}")
                Game(models[i], models[j], models[h]).play_game()

    models.sort(key=lambda x: x.grade, reverse=True)
    for i in range(model_num_index):
        models[model_num_index * 2 + i] = models[i].variation()

model_num_index = 2
model_num = 3 * model_num_index
models = [Model() for _ in range(model_num)]

# Load the most recent model file for all models
# path = get_most_recent_model("D:/Models")
# for model in models:
#     model.load(path)

import time

start_time = time.time()
for _ in range(100):
    loop()
end_time = time.time()
print((end_time - start_time) / 100)

models[0].save(r"D:/Models")
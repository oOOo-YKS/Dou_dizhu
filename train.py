import os
import glob
from intelligence import Model
from game import Game

train_loop = 30
model_num_index = 5


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

def loop():
    for i in range(model_num - 2):
        for j in range(i + 1, model_num - 1):
            for h in range(j + 1, model_num):
                Game(models[i], models[j], models[h], show_process=False).play_game()

    models.sort(key=lambda x: x.grade, reverse=True)
    for i in range(model_num_index):
        models[model_num_index * 2 + i] = models[i].variation()

model_num = 3 * model_num_index
models = [Model() for _ in range(model_num)]

path = get_most_recent_model("D:/Models")
for model in models:
    model.load(path)

from tqdm import tqdm
import time

start_time = time.time()
for _ in tqdm(range(train_loop), desc="Training Progress"):
    loop()
end_time = time.time()
print((end_time - start_time) / train_loop)

models[0].save(r"D:/Models")

import os
import pandas as pd


img_dir = "./data/images"
driving_log_path = "./data/driving_log.csv"

driving_log_data = pd.read_csv(driving_log_path)

img_paths = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith("jpg")]


print(len(img_paths))
print(img_paths[0:4])


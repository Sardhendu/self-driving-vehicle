from src import  commons

train_data_path = "./data/train.p"
valid_data_path = "./data/valid.p"


train_data = commons.read_pickle(train_data_path)
print(train_data.keys())


features = train_data["features"]
labels = train_data["labels"]

print(features.shape)


class Model():
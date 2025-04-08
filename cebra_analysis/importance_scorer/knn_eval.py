import os
import sys
import numpy as np
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
sys.path.append('/home/garullie/CEBRA_analysis/')
from dataset_load import data_load


#index = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
index=10
print(f"SLURM ID = {index}")

with open(f"./embeddings/embeddings_folds.pkl", "rb") as input_file:
    file = pkl.load(input_file)[index]
    embedding = file["embedding"]
    y_train = file["y_train"]

print("opened embedding")

with open(f"./embeddings/embeddings_test_folds.pkl", "rb") as input_file:
    file = pkl.load(input_file)[index]
    test_embedding = file["embedding"]
    y_test = file["y_test"]

print("opened test embedding")
fold_dict = {}

decoder = KNeighborsClassifier(n_neighbors=3, metric = "cosine")
decoder.fit(embedding, y_train)
print("decoding")

for feat in test_embedding.keys():
    prediction = decoder.predict(test_embedding[feat])
    prediction_ = np.array(prediction, dtype=np.int64)
    fold_dict[feat] = {"true":y_test, "prediction": prediction_}

with open(f"./fold_eval/permutations_fold_{index}.pkl", 'wb') as handle:
    pkl.dump(fold_dict, handle)

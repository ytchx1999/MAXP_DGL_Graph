import pandas as pd
from tqdm import tqdm
import pickle

submit = pd.read_csv('../dataset/sample_submission_for_validation.csv')
print(len(submit))

idx_map = {}
for i in tqdm(range(len(submit))):
    idx_map[submit['id'][i]] = i
print(len(idx_map))

dict_path = '../dataset/csv_idx_map.pkl'
with open(dict_path, 'wb') as f:
    pickle.dump(idx_map, f)

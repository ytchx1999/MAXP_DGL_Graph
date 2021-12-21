import pandas as pd
from tqdm import tqdm
import pickle

submit_val = pd.read_csv('../dataset/sample_submission_for_validation.csv')
submit_test = pd.read_csv('../dataset/sample_submission_for_test.csv')
print(len(submit_val))
print(len(submit_test))

idx_map_val = {}
for i in tqdm(range(len(submit_val))):
    idx_map_val[submit_val['id'][i]] = i
print(len(idx_map_val))

idx_map_test = {}
for i in tqdm(range(len(submit_test))):
    idx_map_test[submit_test['id'][i]] = i
print(len(idx_map_test))

dict_path = '../dataset/csv_idx_map_val.pkl'
with open(dict_path, 'wb') as f:
    pickle.dump(idx_map_val, f)

dict_path = '../dataset/csv_idx_map_test.pkl'
with open(dict_path, 'wb') as f:
    pickle.dump(idx_map_test, f)
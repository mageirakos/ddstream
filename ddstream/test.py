import numpy as np
import pandas as pd
import pickle

database = 'noisy_moons'
EXP_PATH = f'shared_data/experiments/{database}_100/'

with open(EXP_PATH+'DETAILS.pkl', 'rb') as f:
    DETAILS = pickle.load(f)

with open(EXP_PATH+'MACRO_CLUSTERS.pkl', 'rb') as f:
    MACRO_CLUSTERS = pickle.load(f)

with open(EXP_PATH+'MICRO_CLUSTERS.pkl', 'rb') as f:
    MICRO_CLUSTERS = pickle.load(f)

with open(EXP_PATH+'MACRO_METRICS.pkl', 'rb') as f:
    MACRO_METRICS = pickle.load(f)

with open(EXP_PATH+'MICRO_METRICS.pkl', 'rb') as f:
    MICRO_METRICS = pickle.load(f)

DETAILS_df = pd.DataFrame.from_dict(DETAILS)
MACRO_CLUSTERS_df = pd.DataFrame.from_dict(MACRO_CLUSTERS)
MICRO_CLUSTERS_df = pd.DataFrame.from_dict(MICRO_CLUSTERS)
MACRO_METRICS_df = pd.DataFrame.from_dict(MACRO_METRICS)
MICRO_METRICS_df = pd.DataFrame.from_dict(MICRO_METRICS)
# dataset_df = pd.read_csv(f'{database}.csv', names=['x-coord','y-coord','label'])

pts_per_batch_series = MICRO_CLUSTERS_df.groupby(['batch id'])['pts'].agg('sum')
total_pts = 0
for batch_num, pts_in_curr_batch in enumerate(pts_per_batch_series):
    print(batch_num, pts_in_curr_batch, total_pts)
    total_pts += pts_in_curr_batch
print(f"total = {total_pts}")
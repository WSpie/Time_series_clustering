import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def time_normalization(time_series, normalize_range=(0, 1)):
    time_series = pd.to_datetime(time_series)
    normalized_ts = time_series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=normalize_range)
    scaler.fit(normalized_ts)
    normalized_ts = scaler.transform(normalized_ts)
    normalized_ts = pd.Series(normalized_ts.flatten())
    return normalized_ts

def data_agg(folder_path):
    fps = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    db = pd.DataFrame()
    for fp in tqdm(fps, total=len(fps)):
        sub_db = pd.read_csv(fp)
        sub_db = sub_db.loc[:, ~sub_db.columns.str.contains('^Unnamed')]
        del sub_db['time']
        if db.empty:
            db = sub_db.copy()
        else:
            db = pd.concat([db, sub_db], axis=0)
    if 'count' in list(db.columns):
        db = db.groupby([col for col in db.columns if not col == 'count'], as_index=False).agg({'count': 'sum'})
    else:
        db = db.groupby([col for col in db.columns if not col == 'cuebiq_id'], as_index=False).agg({'cuebiq_id': 'count'}).rename(columns={'cuebiq_id': 'count'})
    col_elems = [x for x in db.columns if not x in ['date', 'count']]
    new_cols = ['date'] + col_elems + ['count']
    return db[new_cols]
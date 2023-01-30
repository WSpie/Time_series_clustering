import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np

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

def preprocess(db_idx, config):
    thresh = config['thresh']
    db = 'db_' + str(db_idx)
    
    loc_feats = config[db]['loc_feats']
    time_feat = config[db]['time_feat']
    target = config[db]['target']
    
    # parse date
    df = pd.read_csv(config[db]['path'])
    start_date, end_date = df[time_feat].min(), df[time_feat].max()
    date_range = list(map(str, pd.date_range(start_date, end_date).strftime('%Y-%m-%d')))
    
    if os.path.exists(config[db]['agg_path']):
        df_agg = pd.read_csv(config[db]['agg_path'])
    else:
        # aggregate df
        df = pd.DataFrame(df.groupby(loc_feats, as_index=False).agg({time_feat: list, target: list}))
        
        # the new col is to store targets in time-series manner
        def update_val(row, date_range):
            # func used in df apply
            row['val'] = [[] for _ in range(len(date_range))]
            indices = list(map(date_range.index, row[time_feat]))
            for i, idx in enumerate(indices):
                row['val'][idx].append(row[target][i])
            for j in range(len(date_range)):
                if row['val'][j]:
                    row['val'][j] = np.mean(row['val'][j])
                else:
                    row['val'][j] = 0
            row['val'] = np.array(row['val'])
            return row
        
        df['val'] = None
        df = df.apply(update_val, axis=1, args=(date_range,))
        
        # filter out rows with insufficient target data
        df = df[df[target].apply(len) >= thresh]
        
        df_agg = df[[time_feat, 'val']]
        df_agg.to_csv(config[db]['agg_path'], index=False)
    
    return df_agg, date_range
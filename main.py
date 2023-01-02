import os, sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import yaml
from pathlib import Path
from utils.logger import ErrLog
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import warnings
warnings.filterwarnings("ignore")

logger = ErrLog('main')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='kmeans')
    parser.add_argument('--db', type=int, default=1, help='index of the db')
    parser.add_argument('--clean', type=bool, default=False, help='clean all the labels before processing')
    parser.add_argument('--retrain', type=bool, default=False, help='retrain the model even if the results are exist')
    opt = parser.parse_args()
    
    config = yaml.safe_load(Path('config.yaml').read_text())
    data_path = config[f'path_{opt.db}']
    data = pd.read_csv(data_path)
    ts_feats = config[f'feat_{opt.db}']
    
    print('-'*50)
    print(f'Start clustering {data_path}')
    
    try:
        if opt.clean:
            data = data.loc[:, ~data.columns.str.contains('^label_')]
        
        X = data[ts_feats].to_numpy()
        if len(ts_feats) == 1:
            X = X.reshape(-1, 1)
        
        if opt.model == 'kmeans':
            k_start, k_end = config['kmeans_k']
            k_range = range(k_start, k_end+1)
            for n_cluster in tqdm(k_range, total=len(k_range)):
                model = KMeans(n_clusters=n_cluster)
                model.fit(X)
                labels = model.predict(X)
                s_score = silhouette_score(X, labels) # It ranges from -1 to 1, with higher values indicating better clusters.
                print(f'{n_cluster}: {s_score}')
                data[f'label_{n_cluster}'] = labels
                with open(os.path.join('checkpoints', f'{opt.model}_db_{opt.db}_k_{n_cluster}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                data.to_csv(data_path, index=False)
                
        
        print('Clustered peacefully!')
    
    except:
        print(len(ts_feats))
        print('Check error log!')
        logger.exception()
                
    
    
    
    
    
    
    


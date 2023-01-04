import os, sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import yaml
from pathlib import Path
from utils.logger import ErrLog
from utils.plot import plot_time_label
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import warnings
warnings.filterwarnings("ignore")

def model_selection(model_name):
    model_name = model_name.lower()
    return 'kmeans'

logger = ErrLog('main')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='kmeans')
    parser.add_argument('--db', type=int, default=1, help='index of the db')
    parser.add_argument('--retrain', type=bool, default=False, help='retrain the model even if the results are exist')
    opt = parser.parse_args()
    
    # load data and sort by time ASC
    config = yaml.safe_load(Path('config.yaml').read_text())
    data_path = config[f'path_{opt.db}']
    data = pd.read_csv(data_path)
    data = data.sort_values(by='time')
    ts_feats = config[f'feat_{opt.db}'] # determine time series feature (string)
    model_name = model_selection(opt.model)
    
    # make dir for each db checkpoints
    db_cp_dir = os.path.join('checkpoints', f'db_{opt.db}')
    if not os.path.exists(db_cp_dir):
        os.mkdir(db_cp_dir)
    model_dir = os.path.join(db_cp_dir, model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # make dir for each db outputs
    db_out_dir = os.path.join('outputs', f'db_{opt.db}')
    if not os.path.exists(db_out_dir):
        os.mkdir(db_out_dir)
    db_model_dir = os.path.join(db_out_dir, model_name)
    if not os.path.exists(db_model_dir):
        os.mkdir(db_model_dir)
    db_plot_dir = os.path.join(db_model_dir, 'plots')
    if not os.path.exists(db_plot_dir):
        os.mkdir(db_plot_dir)
    
    
    print('-'*50)
    print(f'Start clustering {data_path}')
    
    try:
        score_path = os.path.join(db_model_dir, 'score_dict.pkl')
        if os.path.exists(score_path) and not opt.retrain:
            with open(score_path, 'rb') as dict_f:
                score_dict = pickle.load(dict_f)
            print('Loaded data from cache!')
            
        else:
            # manually clear the cached labels
            if opt.retrain:
                data = data.loc[:, ~data.columns.str.contains('^label_')]
            
            # get unique time
            time = pd.unique(data['time'])
            
            # get corresponding time-sensitive features
            if len(ts_feats) == 1:
                Xs = [data.loc[data['time']==t, ts_feats].to_numpy().reshape(-1, 1) for t in time]
            else:
                Xs = [data.loc[data['time']==t, ts_feats].to_numpy() for t in time]
            
            if model_name == 'kmeans':
                k_start, k_end = config['kmeans_k']
                k_range = range(k_start, k_end+1)
                score_dict = {}
                for n_cluster in tqdm(k_range, total=len(k_range)):
                    # model = KMeans(n_clusters=n_cluster)
                    labels = []
                    scores = []
                    for X in Xs:
                        model = KMeans(n_clusters=n_cluster)
                        model.fit(X)
                        # label = model.predict(X)
                        label = model.labels_
                        label = label[np.argsort(label)]
                        s_score = silhouette_score(X, label)
                        labels.extend(list(label.flatten()))
                        scores.append(s_score)
                    score_dict[f'label_{n_cluster}'] = scores
                    data[f'label_{n_cluster}'] = labels
                    # with open(os.path.join(model_dir, f'{model_name}_db_{opt.db}_k_{n_cluster}.pkl'), 'wb') as f:
                    #     pickle.dump(model, f)
                    data.to_csv(data_path, index=False)
                with open(score_path, 'wb') as dict_f:
                    pickle.dump(score_dict, dict_f)
                print('Clustered peacefully!')
            
        plot_time_label(data, db_plot_dir, score_dict)
        print('Plotted successfully!')
                
        
        
        
    
    except:
        print('Check error log!')
        logger.exception()
                
    
    
    
    
    
    
    


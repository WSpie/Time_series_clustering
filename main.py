import warnings
warnings.filterwarnings("ignore")
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
from utils.models import get_X, use_Kmeans, use_HC, use_DBSCAN
from utils.preprocess import preprocess


logger = ErrLog('main')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='kmeans', help='kmeans, hc')
    parser.add_argument('--db', type=int, default=1, help='index of the db')
    parser.add_argument('--retrain', type=bool, default=False, help='retrain the model even if the results are exist')
    opt = parser.parse_args()
    
    # load config
    config = yaml.safe_load(Path('config.yaml').read_text())
    
    # preprocess data or load agged data
    print(f'Preprocessing db_{opt.db}')
    data, date_range = preprocess(opt.db, config)
    
    # make dir for each db outputs
    db_out_dir = os.path.join('outputs', f'db_{opt.db}')
    if not os.path.exists(db_out_dir):
        os.mkdir(db_out_dir)
    db_model_dir = os.path.join(db_out_dir, opt.model)
    if not os.path.exists(db_model_dir):
        os.mkdir(db_model_dir)
    db_plot_dir = os.path.join(db_model_dir, 'plots')
    if not os.path.exists(db_plot_dir):
        os.mkdir(db_plot_dir)
    
    print('-'*50)
    print(f'Start clustering db_{opt.db}')
    
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
            
            # start clustering
            X = get_X(data)
            
            if opt.model.lower() == 'kmeans':
                score_dict = use_Kmeans(X, config, date_range, db_plot_dir)
            elif opt.model.lower() == 'hc':
                score_dict = use_HC(X, config, date_range, db_plot_dir)
            elif opt.model.lower() == 'dbscan':
                score_dict = use_DBSCAN(X, config, date_range, db_plot_dir)
                
            with open(score_path, 'wb') as dict_f:
                pickle.dump(score_dict, dict_f)
            print('Clustered peacefully!')
                
                
    except:
        print('Check error log!')
        logger.exception()
                
    
    
    
    
    
    
    


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pickle
import itertools
from .plot import plot_vertical

def get_X(df):
    X = None
    for val in df['val']:
        if type(val) == str:
            val = np.array(list(map(float, val.replace('\n', '')[1:-1].split())))
        if X is None:
            X = val.reshape(1, -1)
        else:
            X = np.concatenate((X, val.reshape(1, -1)), axis=0)
    return X

def use_Kmeans(X, config, date_range, db_plot_dir):
    cluster_nums = range(config['clusters'][0], config['clusters'][1]+1)
    s_scores = []
    for k in tqdm(cluster_nums, total=len(cluster_nums), desc='Kmeans'):
        model = KMeans(n_clusters=k)
        model.fit(X)
        labels = model.labels_
        s_score = silhouette_score(X, labels)
        s_scores.append(s_score)
        plot_vertical(X, labels, date_range, k, db_plot_dir, f'{k} Means, score={np.round_(s_score, 4)}')
        
    score_dict = dict(zip(cluster_nums, s_scores))
    return score_dict
    

def use_HC(X, config, date_range, db_plot_dir):
    cluster_nums = range(config['clusters'][0], config['clusters'][1]+1)
    s_scores = []
    for k in tqdm(cluster_nums, total=len(cluster_nums), desc='HC'):
        linked = linkage(X, method='ward')
        labels = cut_tree(linked, n_clusters=k)[:, 0]
        s_score = silhouette_score(X, labels)
        s_scores.append(s_score)
        plot_vertical(X, labels, date_range, k, db_plot_dir, f'{k} Hierarchical clustering, score={np.round_(s_score, 4)}')
        
    score_dict = dict(zip(cluster_nums, s_scores))
    return score_dict

def use_DBSCAN(X, config, date_range, db_plot_dir):
    eps = config['eps']
    min_samples = config['min_samples']
    combinations = list(itertools.product(eps, min_samples))
    
    s_scores = []
    for combo in tqdm(combinations, total=len(combinations), desc='DBSCAN'):
        dbscan = DBSCAN(eps=combo[0], min_samples=combo[1])
        labels = dbscan.fit_predict(X)
        s_score = silhouette_score(X, labels)
        s_scores.append(s_score)
        plot_vertical(X, labels, date_range, -1, db_plot_dir, f'DBSCAN(eps:{combo[0]}, min_samples:{combo[1]}), score={np.round_(s_score, 4)}', combo=combo)
    
        
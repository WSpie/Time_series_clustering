import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
from tqdm import trange
import os

def plot_time_label(df, db_plot_dir, score_dict):
    _, db_idx, model_name, _ = db_plot_dir.split('/')
    df = df.sort_values(by='time')
    labels = [x for x in df.columns if x.startswith('label_')]
    for i in trange(len(labels), desc='Plotting'):
        label = labels[i]
        score = score_dict[label]
        plot_path = os.path.join(db_plot_dir, f'{label}.png')
        
        plt.clf()
        fig, ax = plt.subplots()
        # Create a pivot table to calculate the percentage of each cluster at each time
        table = df.pivot_table(index='time', columns=label, aggfunc='size', fill_value=0)
        # Normalize the table by dividing each value by the sum of all values at that time
        table = table.div(table.sum(axis=1), axis=0)
        # Plot the table as a stacked bar plot
        table.plot.bar(stacked=True, ax=ax)
        ax.set_ylim((0, 1))
        
        # Add Score plot
        ax2 = ax.twinx()
        ax2.autoscale(False)
        ax2.set_ylim((-1, 1))
        ax2.plot(score, label='Score', c='black', marker='o', linewidth=1, alpha=0.5)
        l, lb = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        legend = ax.legend(l + l2, lb + lb2, fontsize=8, ncol=1, handletextpad=0.1, columnspacing=0.2)
        legend.set_zorder = 10
        
        # Round xtick to 4 digits
        xtick_pos = plt.xticks()[0]
        xtick_t = [np.round_(float(t.get_text().split()[-1]), 4) for t in plt.xticks()[1]]
        ax.set_xticks(xtick_pos, xtick_t, rotation=45, fontsize=8)

        plt.title(f'{db_idx} {model_name} {label} score={np.round_(np.mean(score), 4)}')
        ax.set_xlabel('Time (Normalized)', labelpad=10)
        plt.subplots_adjust(bottom=0.25, right=0.85)
        ax.set_ylabel('Percentage', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        plt.savefig(plot_path)

def plot_vertical(X, label, date_range, K, db_plot_dir, desc='', combo=(0, 0)):
    plt.clf()
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    X_min, X_max = np.min(X), np.max(X)
    X_gap = np.abs(X_max - X_min)
    if K == -1: # only for DBSCAN
        unique_label = np.unique(label)
        clusters = len(unique_label)
        fig, axs = plt.subplots(clusters, 1, figsize=(20, 3*clusters))
        for i in range(len(X)):
            lab = label[i]
            if lab != -1:
                axs[lab].plot(X[i], c=colors[lab], alpha=0.05, linewidth=1)
            else:
                axs[clusters-1].plot(X[i], c='k', alpha=0.05, linewidth=1)
        for k in range(clusters):
            axs[k].set_ylim([X_min-0.1*X_gap, X_max+0.1*X_gap])
            axs[k].set_xlim([0, len(date_range)-1])
            axs[k].set_xticks([])
        axs[-1].set_xticks(range(len(date_range)), date_range, rotation=45)
        fig.suptitle(desc, fontsize=20)
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(db_plot_dir, f'eps_{combo[0]}_samp_{combo[1]}.png'))
    else:
        fig, axs = plt.subplots(K, 1, figsize=(20, 3*K))
        for i in range(len(X)):
            lab = label[i]
            axs[lab].plot(X[i], c=colors[lab], alpha=0.05, linewidth=1)
        for k in range(K):
            axs[k].set_ylim([X_min-0.1*X_gap, X_max+0.1*X_gap])
            axs[k].set_xlim([0, len(date_range)-1])
            axs[k].set_xticks([])
        axs[-1].set_xticks(range(len(date_range)), date_range, rotation=45)
        fig.suptitle(desc, fontsize=20)
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(db_plot_dir, f'clusters_{K}.png'))
    
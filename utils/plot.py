import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
from tqdm import trange
import os

def plot_time_label(df, db_plot_dir):
    _, db_idx, model_name, _ = db_plot_dir.split('/')
    df = df.sort_values(by='time')
    labels = [x for x in df.columns if x.startswith('label_')]
    for i in trange(len(labels), desc='Plotting'):
        label = labels[i]
        plot_path = os.path.join(db_plot_dir, f'{label}.png')
        
        plt.clf()
        # Create a pivot table to calculate the percentage of each cluster at each time
        table = df.pivot_table(index='time', columns=label, aggfunc='size', fill_value=0)
        # Normalize the table by dividing each value by the sum of all values at that time
        table = table.div(table.sum(axis=1), axis=0)
        # Plot the table as a stacked bar plot
        table.plot.bar(stacked=True)
        # Round xtick to 4 digits
        xtick_pos = plt.xticks()[0]
        xtick_t = [np.round_(float(t.get_text().split()[-1]), 4) for t in plt.xticks()[1]]
        plt.xticks(xtick_pos, xtick_t, rotation=45, fontsize=8)
        plt.title(f'{db_idx} {model_name} {label}')
        plt.xlabel('Time (Normalized)')
        plt.ylabel('Percentage')
        plt.savefig(plot_path)
        
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_loc_label(data):
    labels_name = [x for x in data.columns if x.startswith('label_')]
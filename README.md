# Time_series_clustering

Our main approaches for Time Series Clustering is KMeans, DWT, etc.

To run the program, type
```bash
python main.py --method kmeans --db 2 --clean True
```

We use Silhoutte Score to evaluate the results of clustering, and it ranges from -1 to 1, with higher values indicating better clusters.

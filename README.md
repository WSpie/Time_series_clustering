# Time_series_clustering

Our main approaches for Time Series Clustering is KMeans, DWT, etc.

To run the program, type
```bash
python main.py --model kmeans --db 2 --retrain True
```
The default model is kmeans and the default dataset is index 1, and retrain is to train the data ignore the existence of cache.

We use Silhoutte Score to evaluate the results of clustering, and it ranges from -1 to 1, with higher values indicating better clusters.

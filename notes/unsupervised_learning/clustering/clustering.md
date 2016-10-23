# Clustering

## Unsupervised Learning

* Data set without labels or of same category

## Clustering Movies

## How Many Clusters?

* K-means clustering

## Match Points with Clusters

* Determine half-space for each cluster point
* Follow steps of 1. Assign data points to cluster centres, 2. Optimize
* Cluster center moved so as of minimize total distance from data points

## K-Means Cluster Visualisation

* [Visualisation](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
* Initial placement of cluster centres is both random and very important 

## Sklearn

* [sklearn.cluster.KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* n_clusters most important parameter

## Limitations of K-Means

* The output for any fixed training set will not always by the same
* K-means is a "hill climbing" algoritm, initial conditions are important
* Leads to counterintuitive clusters

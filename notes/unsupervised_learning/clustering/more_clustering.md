# More Clustering

## Single Linkage Clustering

* Consider each object (data point) a cluster (n objects)
* Define intercluster distance as the distance between the closest two points in the two clusters
* Merge two closest clusters
* Repeat n-k times to make k clusters
* (Also max, average/mean link clustering)

## Running Time of SLC

O(n^3)

## Issues with SLC

* Can lead to unintuitive clusters due to need to minimise distance between clusters

## Soft Clustering

Assume the data was generated by
1. Select one of k Gaussian - fixed, known variance - uniformly
2. Sample x_i from that Gaussian
3. Repeat n times

Find a hypthesis h = <mu_1, ..., mu_k> that maximises the probability of the data (maximum likelihood)

## Maximum Likelihood Gaussian

The ML mean of the Gaussian M is the mean of the data

What if k different means? Use hidden variables

## Expectation Maximisation

Ignore prior due to uniformity assumption

Summation on the left should be with respect to variable j not i

Leads to k-means if cluster assignments use arg-max

## Properties of EM

* Monotonically non-decreasing likelihood
* Does not (guarantee to) converge (practically does)
* Will not diverge
* Can get stuck - random restart
* Works with any distribution (If expectation, maximisation solvable, generally expectation more difficult)

## Clustering Properties

For clustering scheme, P_D

* Richness: For any assignment of object to clusters, there is some distance
  matrix D such that P_D returns that clustering, for all C, exists D s.t. P_D = C
* Scale-invariance: Scaling distances by a positive value does not change the
  clustering for all D, K>0 P_D = P_KD
* Consistency: Shrinking intracluster distances and expanding intercluster
  distances does not change the clustering P_D = P_D'

| Single-link clustering, stop when... | Richness | Scale-invariance | Consistency |
| ------------------------------------ | -------- | ---------------- | ----------- |
| n/2 clusters reached                 |    N     |         Y        |      Y      |
| clusters are theta units apart       |    Y     |         N        |      Y      |
| cluters are theta/omega units apart  |    Y     |         Y        |      N      |

where omega = max_i,j d(i,j)

## Impossibility Theorem (Kleinbery)

No clustering scheme can achieve all three of:
* Richness
* Scale-invariance
* Consistency

## Summary

* Algorithms
  - K-means
  - SLC (terminates fast)
  - EM (soft clusters)
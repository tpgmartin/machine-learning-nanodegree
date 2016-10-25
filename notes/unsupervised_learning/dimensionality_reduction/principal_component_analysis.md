# Principal Component Analysis (PCA)

## Overview

Find new coordinate system for data points via traslation and rotation:
* Find centre of data points in original axes
* Define orthogonal vectors from centre
  - One vector parallel to line of data points
  - One vector perpendicular to line of data points
* Define unit vectors based on orthogonal vectors - an imporance vector 
  defining spread

Can find PCA even for datasets without clear line, but only one axis dominates
in case of clear linear distribution of points

## Measurable vs Latent Features

Example: Given the features of a house, what is its price?

| Measurable          | Latent       |
| ------------------- | ------------ |
| Square footage      | Size         |
| No. rooms           | Size         |
| School ranking      | Neigbourhood |
| Neigbourhood safety | Neigbourhood |

## Compression While Preseving Information

How best to condense our N features to 2, so that we really get to the heart of
the Information?

Unkown number of features, but size and neigbourhood should be underlying all
of them.

What's the most suitable feature selection tool?
- Select KBest (k features to keep)

## Composite Features

Many features, but hypothesize a smaller number of features actually driving
the patterns

Try making a composite feature (principal component) that more directly probes 
the underlying phenomenon

Discuss in context of dimensionality reduction

Principal component - not a regression, use projection of data points onto 
principal component

Principal component of a dataset is the direction that has the largest variance
because this direction retains maximum amount of information from original data

Determine regression from principal components

## Maximal Variance and Information Loss

Information loss proportional to length of perpendicular line from principal
axis and data point

## Information Loss and Principal Components

Projection onto direction of maximal variance minimises distance from old 
(higher-dimensional) data point to its new transformed value. This minimises
information loss.

## PCA for Feature Transformation

PCA can automatically combine features into principal components, which is an
unsupervised - and scalable - technique

Maximum number of PCs = Min(no. features, no. training points)

## Review/Definition of PCA

* Systematised way to transform input features into principal components
* Use principal components as new features
* PCs are directions in data that maximise variance (minimise information loss)
  when you project/compress down onto them
* The more variance of data along a PC, the higher that PC is ranked
* Most variance/most information "first PC"
  Second-most variance (without overlapping with first PC) "second PC"
* Maximum number of PCs = number of input features

## PCA in Sklearn

[sklearn.decomposition.PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## When to Use PCA

* Latent featues driving the patterns in data
* Dimensionality reduction
  - visualise high-dimensional data
  - reduce noise
  - make other algorithms work better because fewer inputs (eigenfaces)

## Selecting a number of principal components

Q: What's a good way to figure out how many PCs to use?

A: Train on differnt number of PCs and see how accuracy responds - cut off when
   it becomes apparent that adding more PCs doesn't buy you much more 
   discrimination

   N.B. Only perform feature selection after PCA


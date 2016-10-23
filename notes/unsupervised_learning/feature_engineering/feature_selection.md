# Feature Selection

## Motivation

* Knowledge discovery: interpretability and insight
* Curse of dimensionality

## Algorithms

For n features, choose m s.t. m <= n
How hard is the problems? Exponential i.e. nCm

## Filtering and Wrapping

###Filtering

Input set of features to algorithm, reutrn fewer features that are passed to 
learning algorithm

###Wrapping

Search over subset of features, pass to learning algorithm, which then reports
success and process cycles round

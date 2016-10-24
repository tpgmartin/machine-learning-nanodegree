# Feature Selection

## Motivation

* Knowledge discovery: interpretability and insight
* Curse of dimensionality

## Algorithms

For n features, choose m s.t. m <= n
How hard is the problems? Exponential i.e. nCm

## Filtering and Wrapping

### Filtering

Input set of features to algorithm, reutrn fewer features that are passed to 
learning algorithm

### Wrapping

Search over subset of features, pass to learning algorithm, which then reports
success and process cycles round

## Speed

### Filtering

Advantages
* Speed - but tend to look at features in isolation

Disadvangtages:
* Ignores the learning problem (ignores bias)

(Decision trees information gain measure form of filtering)

### Wrapping

Advantages
* Takes into account model bias
* Takes into account learning problem

Disadvangtages:
* Very slow

## Wrapping

### Filtering

Doesn't care about the learner, can't take advantage of bias of learner

Considerations

* Information gain
* Variance, entropy
* "Useful" features
* Independent/non-redundant

### Wrapping

* Hill climbing
* Randomised optimisation
* Forward search
* Backward search

## Relevance

* x_i is strongly relevant if removing it degrades Bayes Optimal Classifier 
  (BOC)
  - (Strong features can become less relevant if have copy in data set)
* x_i is weakly relevant if
  - not strongly relevant
  - exists subset of features S s.t. adding x_i to S improves BOC
* x_i is otherwise irrelevant

## Relevence vs Usefulness

* Relevance measures effect of BOC
  - Relevance ~ information
* Usefulness measures effect on a particular predictor
  - Usefulness ~ Error given model/learner

## Summary



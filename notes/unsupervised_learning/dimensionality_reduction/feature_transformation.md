# Feature Transformation

## Introduction

The problem of pre-processing a set of features to create a new (smaller? more
compact?) feature set, while retaining as much (relevant? useful?) information
as possible:

x ~ F^N -> F^M

M<N (usually) due to curse of dimensionality
P^T_X (usually) linear combinations of original features

N.B. rather than feature selection we are returning a linear combination of the
original features, not just a subset of the original features

## Feature Transformation

Example: Ad hoc information retrieval - retrieve subset of documents relevant
for a given query

## What are our features?

Features: words - lots of words

Insufficient indicators:
* Polysemy e.g. "car" can have multiple meanings
  - Can lead to false positives
* Synonomy e.g. "car" and "automobile" mean the same thing
  - Can lead to false negatives

## Words like "Tesla"

Rank words that a related to "car", this will help with synonomy

## Independent Components Analysis (ICA)

For linearly independent variables x_i that map to y_i

Variables are independent of one another s.t.
* I(y_i, y_j) = 0
* I(x.y) is a large as possible

As we have linear combinations of variables, we are not losing any of the 
original information

## Independent Components Analysis Two

Blind source separation problem: Multiple people (hidden variables) speaking
into microphones (observables)

Observables record different linear combinations of hidden variables. ICA tells
us that given all the observable data we can piece together the output of a 
given hidden variable exactly - due to indepence of sources

## Cocktail Party Problem

[Example webpage](http://research.ics.aalto.fi/ica/cocktail/cocktail_en.cgi)
[Independent Component Analysis: Algorithms and Applications](http://mlsp.cs.cmu.edu/courses/fall2012/lectures/ICA_Hyvarinen.pdf)

## Matix

Rows represent a feature in original feature space
Columns represent a sample

"Mutual information": how much one variable allows you to predict another 
variable

In this case concerned about mutual information between the observed data and
our projection onto the hidden variables

## PCA vs ICA

|                            |    PCA     |     ICA    |
| -------------------------- | ---------- | ---------- |
| Mutually orthogonal        |     Y      |      N     |
| Mutually independent       |     N      |      Y     |
| Maximal variance           |     Y      |      N     |
| Maximal mutual information |     N      |      Y     |
| Ordered features           |     Y      |      N     |
| Bag of features            |     Y      |      Y     |
| Blind Source Separation    |     N      |      Y     |
| Directional                |     N      |      Y     |

* Are cases where PCA finds mutually independent projections - when all data is
  Gaussian (this is due to central limit theorem)
* [ICA application](http://www.cc.gatech.edu/~isbell/papers/isbell-ica-nips-1999.pdf)
* Processing faces:
  - PCA is global finds (in order of principal component) brightness, average
    face 
  - ICA is local finds parts of faces e.g. nose, eyes, chin
* ICA locality makes it good for procesing natural scenes. ICA finds edges
* ICA returns topics from documents in ad hoc information retrieval problem
* ICA is good for understand the structure of your data
* Feature transformation allows us to understand structure of data

## Alternatives

Random Component Analysis (RCA)

* Generates random directions
* Projects along axes in those directions
* Given projecting from N to M linear combinations of features
* Captures correlations between features 

The big advantage of RCA is that it is fast

Linear Discriminant Analysis (LDA)

* Finds a projection that discriminates based on the label
* LDA explicitly care about labels unlike other feature transformation methods

## Wrap up

[A survey of dimension reduction techniques](http://computation.llnl.gov/casc/sapphire/pubs/148494.pdf)

PCA is mostly about linear algebra, ICA is mostly about probability

## Outro

### Big Data

Algorithmic challenges from gigantic datasets, even linear too slow

### Deep learing

New techniques for getting signal through multiple layers

### Semi-supervised learing

Example: webpages with information about cities, combine labeled data with 
unlabeled data

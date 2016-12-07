# Machine Learning Engineer Nanodegree
## Capstone Proposal
Tom Martin  
3rd December 2016

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

My project will examine the MNIST database of handwritten digits. This is a very 
well known dataset having attracted a great deal of academic attention since its 
inception. Over the years, it has proved fruitful territory for examining a range
of machine learning classifiers, such as [linear classifiers](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf),
[svm](https://people.eecs.berkeley.edu/~malik/cs294/decoste-scholkopf.pdf), 
[k-nearest neighbours](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E0F3BDC7642FBA1D8E2811526BD0E596?doi=10.1.1.106.3963&rep=rep1&type=pdf)
, and a range of 
[neural network implementations](https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/).


### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem
described should be well defined and should have at least one relevant 
potential solution. Additionally, describe the problem thoroughly such that it
is clear that the problem is quantifiable (the problem can be expressed in
mathematical or logical terms) , measurable (the problem can be measured by
some metric and clearly observed), and replicable (the problem can be
reproduced and occurs more than once).

The capstone should attempt to train and tune a classifier that is able to 
correctly determine the handwritten number from the supplied image. The success 
of the classifier will be measured using    


### Datasets and Inputs

The MNIST dataset contatins 70000 samples of handwritten digits, labelled from 0 to 9. 
These are split into subsamples of 60000 and 10000 for training and testing respectively.
The samples themselved contains have been centred and normalised to a grid size 
of 28-by-28 pixels, with each training entry composed of 784 features, 
corresponding the greyscale level for each pixel. The MNIST dataset in this 
case will be the [MNIST original](http://mldata.org/repository/data/viewslug/mnist-original/)
dataset obtained via the [mldata](http://mldata.org/) repository using 
SciKit-Learn's `datasets.fetch_mldata` method.

On a historical note,this dataset is the result of subsampling the original 
NIST dataset so that is was overall more consistent, and more suitable for machine learning: mixing together the 
original training and testing sets. The samples were collected from a combination of 
American Census Bureau employees and American high school students.

This dataset contains both training and testing samples, so it will be used both to train 
and evaluate the classifier. 

### Solution Statement

I propose using a SVM classifier to train a solution that, with a reasonable 
level of accuracy, correctly map a handwritten sample to the correct digit. 
A supervised classifier should be an appropriate solution to the problem as we 
have training data. There are also a number of academic studies, mentioned 
above, that have had success with SVM classifiers. The trained classifier can 
be evaluated using a confusion matrix, and derived metrics such as f1 score, to 
determine its degree of success. To evaluate the trained model thoroughly, 
k-fold cross validation will be used to get a representative performance score 
of the model. Furthermore, we can consider a number of 
[previous models](http://yann.lecun.com/exdb/mnist/) of the datasets using a 
SVM classifier, which have accuracies around 99%. 

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that 
relates to the domain, problem statement, and intended solution. Ideally, 
the benchmark model or result contextualizes existing methods or known 
information in the domain and problem given, which could then be objectively 
compared to the solution. Describe how the benchmark model or result is 
measurable (can be measured by some metric and clearly observed) with thorough 
detail.

This dataset is very well studied 

Benchmark model SVM classifier with Guassian kernel

[Study](https://cseweb.ucsd.edu/~jmcauley/cse190/reports/fa15/025.pdf)

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to 
quantify the performance of both the benchmark model and the solution model.
The evaluation metric(s) you propose should be appropriate given the context of
the data, the problem statement, and the intended solution. Describe how the 
evaluation metric(s) are derived and provide an example of their mathematical 
representations (if applicable). Complex evaluation metrics should be clearly 
defined and quantifiable (can be expressed in mathematical or logical terms).

The evaluation metric for this problem are given by the error rate percentage of 
previous studies

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a 
solution given the problem. Provide thorough discussion for what strategies
you may consider employing, what analysis of the data might be required before 
being used, or which algorithms will be considered for your implementation. The 
workflow and discussion that you provide should align with the qualities of the
previous sections. Additionally, you are encouraged to include small 
visualizations, pseudocode, or diagrams to aid in describing the project design
, but it is not required. The discussion should clearly outline your intended 
workflow of the capstone project.

This project will follow a typical machine workflow, starting from the dataset 
acquisition, then the model generation, and then an evaluation and optimisation 
process. There will be no preprocessing applied to the dataset for the two 
reasons. The first is that this dataset is already in a form understandable to 
SciKit-Learn, as it will be downloaded using SciKit-Learn `fetch_mldata` 
method. Secondly, to form a meaningful comparison with the benchmark study, we 
want a comparative dataset to begin with and in this case particular study 
performed no preprocessing either. Feature engineering will not be used either 
as the dataset is not permitting of additional feature analysis - the features 
are just a pixel-by-pixel greyscale score.

Due to the previous success of such classifiers and the wealth of related 
former studies, this project will use a SVM classifier. To both optimise and 
evaluate the classifier a test-training split will be done

* No additonal preprocessing of the data will be required
* Use k-fold cross validation for parameter optimisation

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?

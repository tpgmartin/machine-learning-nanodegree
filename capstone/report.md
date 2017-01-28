# Machine Learning Engineer Nanodegree
## Capstone Project
Tom Martin
3rd January 2017

## I. Definition

### Project Overview

My project will examine the MNIST database of handwritten digits. The project 
aims are to deduce an accurate SVM classfier for this dataset and analyse its 
performance against a benchmark study, noted below.

This is a very well known dataset having attracted a great deal of academic 
attention since its inception. More broadly, the analysis of automated 
handwriting recognition has applications for fields where it is important to 
quickly and securely process handwritten documents at scale. For instance, this 
can be useful in processing historical documents, input to handheld devices via 
a stylus or pen, or determining authorship of incriminating documents.

Over the years, it has proved fruitful territory for examining a range
of machine learning classifiers, such as linear classifiers[1], svm[2], 
k-nearest neighbours[3], and a range of neural network implementations[4].
There have therefore been a number of different approaches shown to be suitable 
to classify the dataset correctly. A paper by Hartwick[5] is particularly 
relevant for this projet as he has provided a clear analysis of the 
dataset without any further preprocessing with an SVM classifier. For these 
reasons, this paper will form a benchmark for the following analysis. 

### Problem Statement

The capstone will attempt to train and tune a SVM classifier that is able to 
correctly determine the number intended from the supplied image of a 
handwritten sample. The model produced will be trained, tested and validated 
against the supplied dataset.  The success of the classifier will be measured 
using the Scikit-Learn metric's module `metrics.accuracy_score`,
`metrics.confusion_matrix`. From these metrics we can derive both the error 
rate and per digit error rate in order to enable a direct comparsion with 
the benchmark model below. It should be mentioned that some of these metrics 
are typically used in problems of binary classifaction, but can be generalised 
for an arbitrary number of classes[6]. This is covered in more detail in the 
"Metrics" section.

I propose using a SVM classifier to train a solution that, with a reasonable 
level of accuracy, correctly maps a handwritten sample to the correct digit. 
A supervised classifier should be an appropriate solution to the problem as we 
have training data. There are also a number of academic studies that have had 
success with SVM classifiers[2]. Before building the model, I will use 
principal component analysis (PCA) with dimension reduction. PCA is 
used to reduce the feature space of the training data to reduce the overall 
training and testing time, as well as making it easier to graphically 
illustrate - can produce two-dimensional plot of classifying boundaries. 
This is especially useful for a dataset both as large and feature-rich as the 
MNIST dataset. It has been shown that in general PCA does not negatively 
impact the accuracy of a classifier, and has even been shown to boost the 
accuracy of the SVM classfier[9]. I will evaluate the classfier using two 
different kernels: Gaussian, or RBF kernel, and polynomial. The choice is for 
two reasons: Firstly, Both these kernels are useful in cases such as thus where 
the data set is not linearly separable. Secondly, I want to make a direct 
comparison with the benchmark study, which used the Gaussian kernel. 

The trained classifier can be evaluated using a confusion matrix, and derived 
metrics to determine its degree of success. To evaluate the 
trained model thoroughly, k-fold cross validation will be used to get a 
representative performance score of the model. Furthermore, we can consider a 
number of previous models[8] of the dataset using a SVM classifier, which have 
accuracies around 99%.

### Metrics

The evaluation metrics for this model will be the confusion matrix, accuracy, 
error rater, andper digit error rate. The latter two are to enable a direct 
comparison with the benchmark, which calculates these values in the paper.

I will be focussing on using the confusion matrix and accuracy to evaluate the 
classfier for the two main reasons. Firstly, the labels in the dataset are 
fairly uniformly distributed, as shown in figure 2 below. This means we can take 
the simpler option of just using accuracy as we do not need to consider 
imbalances between classes - accuracy will strongly correspond to other 
measures such as precision. Secondly, this is a multi-class classification 
problem where we are more interested in correct classifactions than 
misclassifications, so accuracy is sufficient to meaningfully evaluate the 
classfier on its own. The confusion matrix will still contain useful 
information especially if there is strong presence of misclassifications on a 
per digit bases, which would not be clear from accuracy alone.

A supervised classfier such as SVM has known labelled data, so we can determine 
the number of true positives, true negatives, false positives, and false 
negatives these are ultimately derived from the confusion matrix. To be clear, 
these terms are defined as follows:

* True positives: Entries that are correctly labelled
* True negatives: Entries that are correctly rejected
* False positives: Entries that a wrongly identified with a given label
* False negatives: Entries for a given label that are wrongly identified with 
other labels

In the general case, a confusion matrix is simply a matrix illustrating the 
mapping from the true labels to the predicted labels. Elements along the 
diagonal represent a correct classification, whereas the off-diagonal represent
a misclassification. A confusion matrix can be a useful check to 
see what digits in particular are most likely confused for one another. From 
here we can derive[6] the accuracy.

Accuracy is given by the total number of correct classifcations, both true 
positives and true negatives divided by the total dataset population. This 
can be given by the following equation,

```math
accuracy = (tp + tn) / (tp + tn + fp + fn)
```

where tp, tn, fp, and fn stand for true positivem true negative, false positive, 
false negative respectively.

From this, the error rate or rate at which the classifier misclassifies can be 
derived,

```math
error rate = 1 - accuracy 
           = 1 - (tp + tn) / (tp + tn + fp + fn) 
           = (fp + fn) / (tp + tn + fp + fn)
```

To determine the error rate for a given digit I refer to the benchmark 
paper[5], which defines the error rate as the ratio of false negatives to the 
sum of true positives and false negatives.

```math
error rate (for given digit) = fn / tp + fn
```

This can be derived from the confusion matrix: For a given row, divide the 
off diagonal entries by the sum of all entries for that row.

These metrics altogether will give us a means to determine how well the 
classifier correctly labels the digits as well the error rate per digit. The 
error rate as well as all the other metrics discussed in this section will be 
calculated using the SciKit-Learn `metrics.accuracy_score`, 
`metrics.confusion_matrix` methods.

## II. Analysis

### Data Exploration

The MNIST dataset contatins 70000 samples of handwritten digits, labelled from 
0 to 9. These are split into subsamples of 60000 and 10000 for training and 
testing respectively. The samples themselve have been centred and normalised 
to a grid size of 28-by-28 pixels, with each training entry composed of 784 
features, corresponding the greyscale level for each pixel. The MNIST 
dataset in this case will be the MNIST original[7] dataset obtained via the 
mldata repository using SciKit-Learn's `datasets.fetch_mldata` method. A sample 
of the dataset is given below.

![Sample of MNIST Dataset](./images/mnist_sample.png "Sample of MNIST Dataset")

The class labels in the testing set are roughly uniformly distributed, with the 
number of occurences of each label ranging from around 6300 and 7900. The 
distribution is shown graphically in figure 1 above. This distribution of 
labels means that no special sampling needs to take place to train and test 
correctly, and as discussed above means we can opt for the simpler choice of 
deriving only the accuracy as the evaluation metric.

On a historical note, this dataset is the result of subsampling the original 
NIST dataset so that is was overall more consistent, and more suitable for 
machine learning: mixing together the original training and testing sets. The 
samples were collected from a combination of American Census Bureau employees 
and American high school students. This dataset contains both training and 
testing samples, so no further data is needed to evaluate the classifier.

### Exploratory Visualization

The plot below illustrates the distribution of labels in the target data, 
showing them to be fairly uniformly distributed. This is useful in relation to 
the choice of evaluation metric, as noted above. Due to this uniformity, we can 
opt for the simpler choice of using just the accuracy as we due not need to 
consider imbalances and skew.

![Frequency of Occurence of Class Labels](./images/frequency_of_occurence_of_class_labels.png "Frequency of Occurence of Class Labels")

### Algorithms and Techniques

This report will train a scalable vector machine (SVM) classifier to correctly 
label the samples. SVM is a way of classifying data sets with a hyperplane or 
boundary bewteen data point clusters. This is achieved by algorithmically 
finding the hyperplane with maximum margin, or in other words finding the line 
that maximises the distance between itself and the points of the data set. SVM 
is particularly powerful when dealing with nonlinear data by employing the 
so-called kernel trick: mapping the input data points to a higher dimensional 
feature space.

The SVM provided by Scikit-Learn's `svm.SVC` module supports the following 
parameters relevant for this investigation,

* C - penalty parameter, how closely hyperplane follows the datapoints
* kernel - the kernel choice, either RBF or polynomial
* degree - the degree of the polynomial kernel
* gamma - kernel coefficient, defines radius of influence for a given datapoint

The specific parameter values will be decided using grid search cross validation 
starting from a set of values following the relevant literature. This technique 
is a simple and effective means of optimising the classifier, as a way of 
justifying our initial assumptions. Other parameters not specified here will be 
used at their default values.

Due to the size of the dataset used in this project it is necessary to use a 
hybrid approach[10]. This means initially running the dataset through a k 
nearest neighbours classifier (KNN) and only run the incorrectly labelled 
digits through the SVM classifier. Together this known as KNN-SVM, and has the 
dual advantage of achieving a high level of accuracy on a feature rich dataset 
using SVM, whilst offloading a large proportion of the classification problem 
to a much cheaper KNN classifier.

The classifier will be trained using a validation set by performing a test 
train split. I will vary the following parameters,

* PCA: `n_components`
* KNN: `n_neighbors`
* SVM: `C`, `degree`, `gamma`, `kernel`

Each trained classifier will be used to evaluate the test set. The performance 
metrics to be used are discussed above.

Preprocessing will be done using PCA with dimension 
reduction. This is discussed at greater length below.

### Benchmark

This dataset is very well studied and as such, there are many comparable 
studies to check against. For the project, I will make direct comparison to 
the paper referenced above by Hartwick[5]. This paper produces results for a 
SVM classifier with Guassian kernel, with parameters,

```math
C = 10^6
gamma = (1/len(features)) * 10^-3.5 (approx. 4 * 10^-7)
```

The model in this paper achieves a error of around 1.4% against the MNIST 
testing set, the paper also gives the per digit error rate. Given all these 
results and the availability of the identical testing set, a direct comparison 
with this paper's results is possible.

However, unlike the benchmark study, I will perform preprocessing on the dataset 
using PCA. Following the study by Lei and Govindaraju[8] I 
will choose to model with several different numbers of principal components 
between 25 and 100, as this is where they found a boosted classifier 
performance. I will also use a hybrid approach, using a combination of KNN and 
SVM classifiers, due to the size of the dataset as discussed above.

## III. Methodology

### Data Preprocessing

The data was fetched using Scikit-Learn's `datasets.fetch_mldata`, which 
creates a local cache for subsequent reads.

Data preprocessing was achieved using PCA with dimensional reduction. This was 
to enable a speed up in the training and execution time of the classifier, 
which has been shown to not degrade performance in general - and has even let 
to some improvement[8]. This is also a very necessary step when processing such 
a large and feature reach dataset such as MNIST especially when training on 
general purpose hardware. The number of components to use in this step comes 
follows from two considerations. Firstly, a previous study using PCA on the 
MNIST dataset showed boosted SVM performance for a number of principal 
components below 100[8]. Secondly, by looking at the explained variance ratio, 
we see that over 91% of the variance is accounted for by the top 100 
components.

```code
pca = PCA().fit(X) # where X is the imported training set
explained_variance = pca.explained_variance_ratio_
sum(explained_variance[101]) # => 91.6%
```

For my implementation, I used the Python `pickle` 
module to persist the target and and preprocessed data samples across files.

PCA was performed before the test train split to ensure consistent 
analysis, the dimensionality of the training data should be the same as the 
test data.

See `Downloading PCA and Caching.ipynb` in the `code_samples` directory for 
implementation details.


### Implementation
In this section, the process for which metrics, algorithms, and techniques 
that you implemented for the given data will need to be clearly documented. It 
should be abundantly clear how the implementation was carried out, and 
discussion should be made regarding any complications that occurred during 
this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the 
given datasets or input data?_
- _Were there any complications with the original metrics or techniques that 
required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated 
functions) that should be documented?_

Reference
* Solution statement
* Evaluation metrics
* Project design

Details

* Using SVM classifier
* Using RBF and polynomial kernels for comparison
* Optimisation: Parameter selection via grid search cross validation 
* Evaluation: k-fold cross validation
* Metrics: accuracy, error, confusion matrix

Cache variables in notebook: https://stackoverflow.com/questions/31255894/how-to-cache-in-ipython-notebook

The SVM classifier trained 

See `Train and Optimise Classifier.ipynb` in the `code_samples` directory for 
implementation details.

### Refinement

The classifier refinement step is bundled as part of the classifier training 
using grid search cross validation against the accuracy metric. Results 
including the parameters for the best performing configuration for a given 
number of principal components as well as the total results by parameter set 
are available in `train.txt` in the `results` directory.

-> Insert intermediate results here
-> Discuss them here

Optimisation was acheived using grid search cross-validation with a 5-fold 
cross-validation splitting strategy measured against the accuracy metric. 
The initial parameter set used for this refinement process were derived from ...

See `Train and Optimise Classifier.ipynb` in the `code_samples` directory for 
implementation details.

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be 
evaluated in detail. It should be clear how the final model was derived and why 
this model was chosen. In addition, some type of analysis should be used to 
validate the robustness of this model and its solution, such as manipulating 
the input data or environment to see how the model’s solution is affected (this 
is called sensitivity analysis). Questions to ask yourself when writing this 
section:
- _Is the final model reasonable and aligning with solution expectations? Are 
the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the 
model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) 
in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

| k | PC | Parameter Set | Test Set Accuracy |
| - | -- | ------------- | ----------------- |
| 2 | 50 | 1             | 0.976533006836    |
| 3 | 50 | 2             | 0.979185797368    |
| 4 | 50 | 2             | 0.980206101418    |
| 5 | 50 | 2             | 0.980716253444    |

| k | PC  | Parameter Set | Test Set Accuracy |
| - | --- | ------------- | ----------------- |
| 5 | 25  | 2             | 0.979491888583    |
| 5 | 50  | 2             | 0.980716253444    |
| 5 | 100 | 2             | 0.978879706152    |

Parameter sets

| 3 | [(1, 3, 0.001, 'poly'), (1, 3, 0.001, 'rbf'), (1, 3, 0.0001, 'rbf'), (1, 4, 0.0001, 'rbf'), (1, 5, 0.0001, 'rbf'), (1, 6, 0.0001, 'rbf'), (10, 3, 0.0001, 'rbf'), (10, 4, 0.0001, 'rbf'), (10, 5, 0.0001, 'rbf'), (10, 6, 0.0001, 'rbf'), (100, 3, 0.0001, 'rbf'), (100, 4, 0.0001, 'rbf'), (100, 5, 0.0001, 'rbf'), (100, 6, 0.0001, 'rbf'), (1000, 3, 0.0001, 'rbf'), (1000, 4, 0.0001, 'rbf'), (1000, 5, 0.0001, 'rbf'), (1000, 6, 0.0001, 'rbf')] |
| 2 | [(1, 3, 0.001, 'poly'), (1, 3, 0.0001, 'poly'), (10, 3, 0.001, 'poly'), (10, 3, 0.0001, 'poly'), (100, 3, 0.001, 'poly'), (100, 3, 0.0001, 'poly'), (1000, 3, 0.001, 'poly'), (1000, 3, 0.0001, 'poly')] |
| 1 | [(1, 3, 0.001, 'poly'), (1, 3, 0.0001, 'poly'), (1, 3, 0.0001, 'rbf'), (1, 4, 0.0001, 'rbf'), (1, 5, 0.0001, 'rbf'), (1, 6, 0.0001, 'rbf'), (10, 3, 0.0001, 'rbf'), (10, 4, 0.0001, 'rbf'), (10, 5, 0.0001, 'rbf'), (10, 6, 0.0001, 'rbf'), (100, 3, 0.0001, 'rbf'), (100, 4, 0.0001, 'rbf'), (100, 5, 0.0001, 'rbf'), (100, 6, 0.0001, 'rbf'), (1000, 3, 0.0001, 'rbf'), (1000, 4, 0.0001, 'rbf'), (1000, 5, 0.0001, 'rbf'), (1000, 6, 0.0001, 'rbf')] |

See `Evaluate Classifier.ipynb` in the `code_samples` directory for 
implementation details.

### Justification

The final classifier used an arbitrary selection of parameters from paramter 
set one.

The benchmark reported above produced a SVM classifier with an error of 1.4%. 
This compares favourably with the classfier trained in this project with has an 
error of 1.93% (corresponding to an accuracy of 98.07%). Discrepancies 
between the benchmark results may be due to,

* Preprocessing with PCA on the intial dataset, the benchmark study did not use 
any preprocessing
* Combination of KNN and SVM classifiers, the benchmark study trained the SVM 
directly on the MNIST set
* Different parameters used in final classifier, potentially related to 
previous two points

More detailed comparision with the benchmark follows by comparing the error 
per digit. This gives some indication of how the two classifiers differ and 
highlight the effect of preprocessing. Accounting for the overall larger error 
for the classifier trained in this project in relative terms, the error per 
digit was fairly similar dispite the preprocessing. In both cases the digits 1, 
0, 6 has the lowers error rates, and digit 9 had siginficantly greater error 
rates. Interesting differences between the classifiers are highlighted by 
looking at the error rate for digit 3, which had the 4th lowest error rate in 
the benchmark, but had the second worst error rate in my trained classifier. 
It is not clear how this relates to PCA and KNN.

| Digit | Benchmark Error | Error per Digit | 
| ----- | --------------- | --------------- | 
| 0	    | 0.0061	        | 0.0076	        | 
| 1	    | 0.0052	        | 0.0062	        | 
| 2	    | 0.0164          | 0.0135	        | 
| 3	    | 0.0148	        | 0.0313	        | 
| 4	    | 0.0132	        | 0.0236	        | 
| 5	    | 0.0168	        | 0.0288	        | 
| 6	    | 0.0104	        | 0.0101	        | 
| 7	    | 0.0184	        | 0.0198	        | 
| 8	    | 0.0184	        | 0.0219	        | 
| 9	    | 0.0277	        | 0.0319	        | 


## V. Conclusion

### Free-Form Visualization

The results for this project can be powerfully illustrated using a confusion 
matrix shown below. This is from the results of the optimised classifier 
detailed in the section "Justification" above.

![Predicted vs True label](./images/confusion_matrix.png "Predicted vs True label")

This demonstrates a very powerful performance overall from the classifier, as 
well as highlighting the digits that proved more difficult relative to others. 
This is discussed in more detail above.

### Reflection

The workflow for this project was as follows,

1. Research of problem area and relevant benchmark in the literature
2. Finding and downloading relevant dataset
3. Preprocessed dataset using PCA
4. Initial classification of dataset using KNN
5. Trained SVM using range of paramters - only used samples not correctly labelled by KNN
6. Produced more results for most successful KNN-SVM classifier

The most difficult stages in the project were also happily the most 
interesting, those being steps 4 and 5. I was not clear in my previous 
research and project proposal how to efficiently run the SVM classifier against 
such a large dataset. This proved to be a real bottleneck for the project and 
required further investigation to proceed. However the hybrid approach used 
was both very effective and also a first for me.

The final model is robust, effective and efficient. In this project I have been 
able to derive a classifier with over 98% accuracy, highly competitive with 
the benchmark study, in a total running time of only 48.6s (on my laptop). 
This is extremely promising moving into a more general setting.

### Improvement

The major bottleneck for this project was the implementation of the SVM 
classifier used by Scikit-Learn. This SVM does not scale well for larger 
datasets, with a reported order of growth between the square of the number of 
samples and the cube the number of samples[11]. A number of solutions to 
explore include trying to better utilise caching of calculated distances in 
memory, trying alternatives to Scikit-Learn like Vowpal Wabbit, or make better 
use of dedicated hardware, such as running the SVM on the GPU rather than a 
general purpose CPU. If a significant performance improvement can be made for 
the SVM classifier then I believe a future classifier could outperform the 
model created in this project. This is because a more opitimised classifier 
would be less reliant on preprocessing or hybrid approaches.

### References

[1] http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf "Gradient-Based Learning Applied to Document Recognition"

[2] https://people.eecs.berkeley.edu/~malik/cs294/decoste-scholkopf.pdf "Training Invariant Support Vector Machines"

[3] http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E0F3BDC7642FB
A1D8E2811526BD0E596?doi=10.1.1.106.3963&rep=rep1&type=pdf "Deformation Models for Image Recognition"

[4] https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/ "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"

[5] https://cseweb.ucsd.edu/~jmcauley/cse190/reports/fa15/025.pdf "Reproducing Results of Guassian Kernel SVM classifiers on the MNIST Dataset"

[6] http://softclassval.r-forge.r-project.org/2013/2013-01-03-ChemomIntellLabSys
tTheorypaper.html "Validation of Soft Classification Models using Partial Class Memberships: An Extended Concept of Sensitivity & Co. applied to Grading of Astrocytoma Tissues"

[7] http://mldata.org/repository/data/viewslug/mnist-original/ "MNIST (original)"

[8] https://www.researchgate.net/profile/Giovanni_Felici/publication/226795010
_Feature_Selection_for_Data_Mining/links/53e413d70cf25d674e94b475.pdf#p
age=78  "Speeding Up Multi-class SVM Evaluation by PCA and Feature Selection"

[9] http://yann.lecun.com/exdb/mnist/ "The MNIST Database of Handwritten Digits"

[10] https://www.vision.caltech.edu/Image_Datasets/Caltech101/nhz_cvpr06.pdf SVM-KNN: Discriminative Nearest Neighbor Classification for Visual Category Recognition

[11] http://scikit-learn.org/stable/modules/svm.html#complexity "Support Vector Machines - Complexity"
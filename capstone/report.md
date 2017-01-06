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
`metrics.confusion_matrix`, and `metrics.recall_score`. In particular from the 
recall rate we can derive the error rate in order to enable a direct comparsion with the benchmark 
model below. It should be mentioned that some of these metrics are 
typically used in problems of binary classifaction, but can be generalised for 
an arbitrary number of classes[6]. This is covered in more detail in the 
"Metrics" section.

I propose using a SVM classifier to train a solution that, with a reasonable 
level of accuracy, correctly maps a handwritten sample to the correct digit. 
A supervised classifier should be an appropriate solution to the problem as we 
have training data. There are also a number of academic studies that have had 
success with SVM classifiers[2]. Before building the model, I will use 
principal component analyis (PCA) with dimension reduction. PCA is 
used to reduce the feature space of the training data to reduce the overall 
training and testing time, as well as making it easier to graphically 
illustrate - can produce two-dimensional plot of classifying boundaries. 
This is especially useful for a dataset both as large and feature-rich as the 
MNIST dataset. It has been shown that in general PCA does not negatively 
impact the accuracy of a classifier, and has even been shown to boost the 
accuracy of the SVM classfier[9]. I will evaluate the classfier using two 
different kernels: Gaussian, or RBF kernel, and polynomial. The choice is for 
two reasons: Firstly, Both these kernels are useful in cases such as thus where 
the data set is not linearly separable. Secondly, I want to make a direct comparison 
with the benchmark study, which used the Gaussian kernel.

The trained classifier can be evaluated using a confusion matrix, and derived 
metrics to determine its degree of success. To evaluate the 
trained model thoroughly, k-fold cross validation will be used to get a 
representative performance score of the model. Furthermore, we can consider a 
number of previous models[8] of the dataset using a SVM classifier, which have 
accuracies around 99%.

### Metrics
In this section, you will need to clearly define the metrics or calculations 
you will use to measure performance of a model or result in your project. These 
calculations and metrics should be justified based on the characteristics of 
the problem and problem domain. Questions to ask yourself when writing this 
section:
- _Are the metrics you’ve chosen to measure the performance of your models 
clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on 
the problem and solution?_

The evaluation metrics for this model will be the confusion matrix, accuracy, 
and error rate. The latter is to enable a direct comparison with the 
benchmark, which calculates this value in the paper. We noted above that the 
error rate can be derived from the recall rate.


General reasoning why these metrics???
-> Do I want to remove the confusion matrix?

Error: 1 - accuracy meaning?

I will be focussing on using the confusion matrix and accuracy to evaluate the 
classfier for the two main reasons. Firstly, the labels in the dataset are 
fairly uniformly distributed, as shown in figure 2 below. This means we can take 
the simpler option of just using accuracy as we do not need to consider 
imbalances between classes - accuracy will map to other measures such as 
precison. Secondly, this is a multi-class classification problem where we 
are more interested in correct classifactions than misclassifications, so 
accuracy is sufficient to meaningfully evaluate the classfier on its own.

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
here we can derive[6] both the accuracy and recall.


Accuracy is given by the total number of correct classifcations, both true 
positives and true negatives divided by the total dataset population. This 
can be given by the following equation,

```math
accuracy = (tp + tn) / (tp + tn + fp + fn)
```

where tp, tn, fp, and fn stand for true positivem true negative, false positive, 
false negative respectively.

Recall is the result of dividing the true positives by the sum of true 
positives with false negatives. This can be given as follows,

```math
recall = tp / (tp + fn)
```

where tp and fn stand for true positive and false negative respectively.
From this, the error rate or rate at which the classifier wrongly 
classifies the samples can be derived,

```math
error rate = 1 - (tp / (tp + fn)) = fn / (tp + fn)
```

which is is also known as the false negative rate. The accuracy and error rate 
per digit could be derived similarly, but on a digit-by-digit basis.

These metrics altogether will give us a means to determine how well the 
classifier correctly labels the digits as well the error rate per digit. The 
error rate as well as all the other metrics discussed in this section will be 
calculated using the SciKit-Learn `metrics.accuracy_score`, 
`metrics.confusion_matrix`, and `metrics.recall_score`. methods.

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

The classifier will be evaluated using the process of k-fold cross validation 
to achieve a representative performance score of the model. This is a process 
of systematically cycling through different training and validation subsets of 
the initial training data and averaging the result. This is to ensure that we 
account for bias in any one validation set. The performance metrics to be used 
are discussed above.

Preprocessing will be done using principal component analyis with dimension 
reduction. This is discussed at greater length below.

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or 
threshold for comparing across performances obtained by your solution. The 
reasoning behind the benchmark (in the case where it is not an established 
result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_

-> What metrics from benchmark? Is an accuracy given?
"The model in this paper achieves a error of around 1.4% against the MNIST 
testing set."
-> Also mention study that uses compression techniques

This dataset is very well studied and as such, there are many comparable 
studies to check against. For the project, I will make direct comparison to 
the paper referenced above by Hartwick[5]. This paper produces results for a 
SVM classifier with Guassian kernel, with parameters,

```math
C = 10^6
gamma = (1/len(features)) * 10^-3.5 (approx. 4 * 10^-7)
```

The model in this paper achieves a error of around 1.4% against the MNIST 
testing set. Given all these results and the availability of the identical 
testing set, a direct comparison with this paper's results is possible.

This project will follow a typical machine workflow, starting from the dataset 
acquisition, then preprocessing, model generation, and then an evaluation and 
optimisation process.Preprocessing will be done using SciKit-Learn's `decomposition.PCA` method. 
Following the study by Lei and Govindaraju[8] I will choose to model with 
several different numbers of principal components between 10 and 100, as this 
is where they found a boosted classifier performance. 

Due to the previous success of such classifiers and the wealth of related 
former studies, this project will use a SVM classifier. To both optimise and 
evaluate the classifier a test-training split will be done on both the training 
lables and test labels. I would like to recover some of the same results as the 
benchmark study i.e. SVM with Guassian kernel, as well as another kernel for 
comparison. Using the training and testing data, I will use grid search 
cross-validation during the optimisation step. Discussed at greater length 
above, I will derive recall, precision and the confusion matrix to determine 
the accuracy of the classifier. I will also use the error rate or false 
negative rate to both determine the accuracy of the derived models and make 
a direct comparison with the benchmark study. 

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly 
documented, if any were necessary. From the previous section, any of the 
abnormalities or characteristics that you identified about the dataset will be 
addressed and corrected here. Questions to ask yourself when writing this 
section:
- _If the algorithms chosen require preprocessing steps like feature selection 
or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or 
characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

Reference
* Solution statement
* Project design

Using PCA with dimension reduction for feature transformation, this is to boost 
the runtime efficiency of the classifier.

Why use PCA?
* Reduce algorithm execution time
* Make graphical representation of analysis simpler
* Preprocessing with PCA `decomposition.PCA`, chose several different numbers 
of principal components between 10 and 100


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

### Refinement
In this section, you will need to discuss the process of improvement you made 
upon the algorithms and techniques you used in your implementation. For 
example, adjusting parameters for certain models to acquire improved solutions 
would fall under the refinement category. Your initial and final solutions 
should be reported, as well as any significant intermediate results as 
necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques 
were used?_
- _Are intermediate and final solutions clearly reported as the process is 
improved?_

Check what can be taken out of Implementation section above

Optimisation with grid search cross validation


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of 
the implementation you designed could be improved. As an example, consider ways 
your implementation can be made more general, and what would need to be 
modified. You do not need to make this improvement, but the potential solutions 
resulting from these changes are considered and compared/contrasted to your 
current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or 
techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know 
how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even 
better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure 
similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a 
clear, concise and specific fashion? Are there any ambiguous terms or phrases 
that need clarification?
- Would the intended audience of your project be able to understand your 
analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal 
grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly 
commented?
- Does the code execute without error and produce results similar to those 
reported?

### References

[1] http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf "Gradient-Based Learning Applied to Document Recognition"

[2] https://people.eecs.berkeley.edu/~malik/cs294/decoste-scholkopf.pdf "Training Invariant Support Vector Machines"

[3] http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E0F3BDC7642FB
A1D8E2811526BD0E596?doi=10.1.1.106.3963&rep=rep1&type=pdf "Deformation Models for Image Recognition"

[4] https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/ "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"

[5] https://cseweb.ucsd.edu/~jmcauley/cse190/reports/fa15/025.pdf "Reproducing Results of Guassian Kernel SVM classifers on the MNIST Dataset"

[6] http://softclassval.r-forge.r-project.org/2013/2013-01-03-ChemomIntellLabSys
tTheorypaper.html "Validation of Soft Classification Models using Partial Class Memberships: An Extended Concept of Sensitivity & Co. applied to Grading of Astrocytoma Tissues"

[7] http://mldata.org/repository/data/viewslug/mnist-original/ "MNIST (original)"

[8] https://www.researchgate.net/profile/Giovanni_Felici/publication/226795010
_Feature_Selection_for_Data_Mining/links/53e413d70cf25d674e94b475.pdf#p
age=78  "Speeding Up Multi-class SVM Evaluation by PCA and Feature Selection"

[9] http://yann.lecun.com/exdb/mnist/ "The MNIST Database of Handwritten Digits"
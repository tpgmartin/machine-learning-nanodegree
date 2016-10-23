# Feature Scaling

Comparing features with different scales - rescale values for different 
features for meaningful comparisons, in range [0,1]

## Feature Scaling Formula

x' = (x - x_min) / (x_max - x_min)

where x' is new (rescaled) feature

## Min/Max Scaler in Sklearn

[Preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html)

## Algorithms Requiring Rescaling

Algorithms that would be affected by feature rescaling:
* SVM with RBF kernel
* K-Means clustering

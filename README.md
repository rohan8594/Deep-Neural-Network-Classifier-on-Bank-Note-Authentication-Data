# DNN Classifier on Bank Note Authentication Data

For this project I am using the [Bank Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) from the UCI repository.

The data consists of 5 columns:

* variance of Wavelet Transformed image (continuous)
* skewness of Wavelet Transformed image (continuous)
* curtosis of Wavelet Transformed image (continuous)
* entropy of image (continuous)
* class (integer)

Where class indicates whether or not a Bank Note was authentic.

In this project I'm going to build a Deep Neural Network Classifier to predict if a Bank note is authentic or not using Contrib.learn

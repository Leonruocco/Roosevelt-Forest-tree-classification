# Roosevelt-Forest tree classification
Multi-class classification of high-dimensionality, many-sample dataset using knn-classifiers and tree-based classifiers.

In this project we train a variety of ML models, suited towards classification tasks, on the Roosevelt Forest Dataset which contains labelled data of tree type observations. The purpose of this classification excercise is to predict the type of tree based upon its attributes. 

www.https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset

In this project we learn about the importance of pre-processing your data as well as careful selection of models in order to achieve high prediction accuracies, avoid overfitting training data and maintaining interpretability of our models. We also see the importance of performing k-fold cross-validation, especially on models trained on relatively small datasets as well as optimising hyperparameters towards improving model accuracy on unseen data and avoiding overfitting.

One of the main take-home messages from this project was appreciating the importance of simplicity and interpretability in model selection. Sometimes the 'simpler the better' is the best approach, and we see this in particular in our exploration of the the Roosevelt Forest dataset. We'll see how even a simple k-nearest-neighbors classifier can perform remarkbly well, achieving high testing accuracy scores while maintaing interpretability such that feature importance can be well evaluated. We go on to perform some feature engineering as well as evaluating the models performance for various k-values. We see that the simpler models, with higher k-values, produce more consistent results despite slightly lower training accuracies.

We also apply some more advanced tree based classifier models, such as random-forests and gradient-boosted forest models and see that while a similarly high prediciton accuracy is achieved, they still underperform the Knn-classifier in this case.

# Data exploration and Knn-classifer 
Notebook: Roosevelt_forest_cover_knnClassifier
In this notebook we analyse a dataset containing tree observations from four areas of the Roosevelt National Forest. This is a large dataset with over 500k entries and a large number of features (55). 

The dataset is very clean and requires no pre-processing clean-up so we jump straight in to EDA in terms of studying the feature distributions. There are both continuous numeric data distributuions and a large number of already one-hot-encoded categorical features. There is up to a 5-order-of-magnitude difference in number of entries between feature columns and a large range of standard deviations indicating that some form of principle-component-analysis will be applicable here.

We take a simple approach at first and simply remove the columns with a standard-deviation less than 0.3, the reasoning being that; high variance data contains more information producing more trainable models. This also helps substantially reduce runtimes. The feature distributions are also scaled with normalisation and standardisation for later comparison.

Analysing the results of a Knn-classifier on all columns with std-devs>0.3, compared to just the continuous numeric data columns, suggests that the inclusion of the extra high-variance categorical feature columns adds sufficient trainability to the model with higher testing accuracies. k-fold cross-validation (CV) confirms this.

Interestingly, we find that the testing accuracies for CV performed on the raw data actually produces the highest testing accuriacies, with the normaliston scaled data performing better than the standardised data---as one might expect given the skewed nature of the distribtuions. Nevertheless, the performance of the unscaled data is surprising and bears closer inspection. 

This is further explored by varying the number of k-neighest neighbors. The scaled data appears to outperform the raw data for higher k-values, but the other way around for low-k values, suggesting this discrepancy is likely due to an overfitting problem as low-k value models are more complex. In the interest of producing generalisable models, one is inlined to choice a high k-value here for a simpler model that is less likely to overfit. 

All-in-all the highest testing accuracy achieved, with CV, was 0.97%.

# Tree-based models

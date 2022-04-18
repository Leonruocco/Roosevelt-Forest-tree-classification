# Classification
Multi-class classification of various datasets using NNs, decision tress and SVMs

In this project we train a variety of ML models, suited towards classification tasks, on different datasets. We learn about the importance of pre-processing your data as well as careful selection of models in order to achieve high prediction accuracies, avoid overfitting training data and maintaining interpretability of our models. We also see the importance of performing k-fold cross-validation, especially on models trained on relatively small datasets as well as optimising hyperparameters towards improving model accuracy on unseen data and avoiding overfitting.

One of the main take-home messages from this project was appreciating the importance of simplicity and interpretability in model selection. Sometimes the 'simpler the better' is the best approach, and we see this in particular in our exploration of the the Roosevelt Forest dataset. We find that a simple Knn-classifier performs remarkably well, achieving high testing accuracy scores while maintaing interpretability such that feature importance can be well evaluated. 

To begin with, we start out with a simple classification training excercise using the very small Palmer-Penguins dataset and see the futility of using overly-complex models like neural-networks to train on such datasets. 

We then take a more serius approach to the much larger Roosevelt Forest dataset, and see how even a simple k-nearest-neighbors classifier can perform remarkbly well. We go on to perform some feature engineering as well as evaluating the models performance for various k-values. We see that the simpler models, with higher k-values, produce more consistent results despite slightly lower training accuracies.

# Palmer-penguins dataset
The first dataset we will work with is the Palmer Island Penguins dataset which contains size measurments for three different penguin species observed on three islands in the Palmer Islands, Antartica. Some information about the dataset can be found here
https://allisonhorst.github.io/palmerpenguins/articles/intro.html
although the contents of the webpage pertain to analysis in R. 

The first notebook we use here is 'Penguin_classification_explore.ipynb'. Here we explore the dataset, do some data preprocessing and fit a neural network. We also compare some different data preprocessing scaling methods. 

# Data exploration with Penguin_classification_explore.ipynb
We begin with some data exploration to get a feel for what our labels and features are. We evidently have a target label of 'species' with 6 features to use for classification. The features take on numeric and categorical values and will need some data pre-processing to prepare for feeding in to a model.

We also notice the limited size of the dataset so we attempt to mitigate this (however futile this may be) by oversampling it by a factor of 3. 

The categorical labels are strings that need to be encoded (ML models need numeric values as inputs). We one-hot-encode the various columns using the ColumnTransformer class as this will create additional binary-classified columns for each category removing the issue of erroneous feature scaling in the model training process. This is done to both the categorical target labels as well as the categorical feature columns.

The next important step is to split our data in to test and train sets. This should be done early on, before any feature scaling, as one runs the risk of 'information leakage' otherwise. The issue being that any feature scaling done to the entire dataset first will encode information pertaining to the training data in the testing data, once the subsequent test/train splitting is done. This can lead to the model overfitting training data and performing poorly on unseen data.

We can then re-scale the remaining non-categorical, numerical feature columns so as to avoid further erroneous feature scaling. We initially try a normalisation approach to scaling since a quick inspection of the feature data suggests non-Gaussian distributions which usually lends itself towards a normalisation approach. This will turn out to not be the case as we'll see.

Once our data is fully pre-processed and ready to be used for training a model we define a simple dense neural netowrk model in TensorFlow, using a standard optimiser and a catgegorical cross-entropy loss functions since we have a multi-class classification problem.

After training the model we inspect the models performance plotting the training and validation losses. Over 50 epochs we see a smooth continuous loss to both training and validation losses suggesting the model is performing well and not overfitting. However, the validation accuracy approaches a 100% rapidly, which is suspicious and suggests overfitting regardless. This could be due to the limited size and scope of the data however. For now we'll move on.

Looking at the confusion matrix we can see the the Adelie penguin class is most accurately predicted. 

We also train separate models on the standardised data as wel as the raw, un-scaled data also for comparions. Looking at the three RMSEs we see that the models trained on scaled data perform much better than the raw data trained model, with the standardisation method wining out albeit with a suspicously perfect RMSE of 0. This bears closer inspection.

# Cross validation and grid-searching hyperparameters with Penguin_classification_NN.ipynb

The Palmer penguin dataset is very small and so, in this file we perform cross-validation (CV) on our neural-network model trained on the penguin dataset to attempt to get a better idea of how well it will perform on un-seen data. The results of a stratified k-fold CV suggests the model still performs quite well, with a 99% accuracy. 

We also perform hyperparameter tuning with GridSearchCV to test a few parameters such as batch size, epoch number and optimisers. Using the results of this search we then train a NN and test it on our hold-out set. 

The NN is still performing suspiciously well, so we're going to take a bit of a tangent here and try out a different dataset.

# Roosevelt Forest dataset: Roosevelt_forest_cover_knnClassifier
In this notebook we analyse a dataset containing tree observations from four areas of the Roosevelt National Forest. In contrast to the penguin dataset,
this is a very large dataset with over 50k entries and a large number of features (55). 

The dataset is very clean and requires no pre-processing clean-up so we jump straight in to EDA in terms of studying the feature distributions. There are both continuous numeric data distributuions and a large number of already one-hot-encoded categorical features. There is up to a 5-order-of-magnitude difference in number of entries between feature columns and a large range of standard deviations indicating that some form of principle-component-analysis will be applicable here.

We take a simple approach at first and simply remove the columns with a standard-deviation less than 0.3, the reasoning being that; high variance data contains more information producing more trainable models. This also helps substantially reduce runtimes. The feature distributions are also scaled with normalisation and standardisation for later comparison.

Analysing the results of a Knn-classifier on all columns with std-devs>0.3, compared to just the continuous numeric data columns, suggests that the inclusion of the extra high-variance categorical feature columns adds sufficient trainability to the model with higher testing accuracies. k-fold cross-validation (CV) confirms this.

Interestingly, we find that the testing accuracies for CV performed on the raw data actually produces the highest testing accuriacies, with the normaliston scaled data performing better than the standardised data---as one might expect given the skewed nature of the distribtuions. Nevertheless, the performance of the unscaled data is surprising and bears closer inspection. 

This is further explored by varying the number of k-neighest neighbors. The scaled data appears to outperform the raw data for higher k-values, but the other way around for low-k values, suggesting this discrepancy is likely due to an overfitting problem as low-k value models are more complex. In the interest of producing generalisable models, one is inlined to choice a high k-value here for a simpler model that is less likely to overfit. 

All-in-all the highest testing accuracy achieved, with CV, was 0.97%.

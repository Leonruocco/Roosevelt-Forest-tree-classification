# Classification
Multi-class classification of various datasets using NNs, decision tress and SVMs

In this project we train a variety of ML models, suited towards classification tasks, on different datasets. We learn about the importance of pre-processing your data as well as careful selection of models in order to achieve high prediction accuracies, avoid overfitting training data and maintaining interpretability of our models. 

# Palmer-penguins dataset
The first dataset we will work with is the Palmer Island Penguins dataset which contains size measurments for three different penguin species observed on three islands in the Palmer Islands, Antartica. Some information about the dataset can be found here
https://allisonhorst.github.io/palmerpenguins/articles/intro.html
although the contents of the webpage pertain to analysis in R. 

The first notebook we use here is 'Penguin_classification.ipynb'.

We beging with some data exploration to get a feel for what our labels and features are. We evidently have a target label of 'species' with 6 features to use for classification. The features take on numeric and categorical values and will need some data pre-processing to prepare for feeding in to a model.

We also notice the limited size of the dataset so we attempt to mitigate this (however futile this may be) by oversampling it by a factor of 3. 

The categorical labels are strings that need to be encoded (ML models need numeric values as inputs). We one-hot-encode the various columns using the ColumnTransformer class as this will create additional binary-classified columns for each category removing the issue of erroneous feature scaling in the model training process. This is done to both the categorical target labels as well as the categorical feature columns.

The next important step is to split our data in to test and train sets. This should be done early on, before any feature scaling, as one runs the risk of 'information leakage' otherwise. The issue being that any feature scaling done to the entire dataset first will encode information pertaining to the training data in the testing data, once the subsequent test/train splitting is done. This can lead to the model overfitting training data and performing poorly on unseen data.

We can then re-scale the remaining non-categorical, numerical feature columns so as to avoid further erroneous feature scaling. We initially try a normalisation approach to scaling since a quick inspection of the feature data suggests non-Gaussian distributions which usually lends itself towards a normalisation approach. This will turn out to not be the case as we'll see.

Once our data is fully pre-processed and ready to be used for training a model we define a simple dense neural netowrk model in TensorFlow, using a standard optimiser and a catgegorical cross-entropy loss functions since we have a multi-class classification problem.

After training the model we inspect the models performance plotting the training and validation losses. Over 50 epochs we see a smooth continuous loss to both training and validation losses suggesting the model is performing well and not overfitting. However, the validation accuracy approaches a 100% rapidly, which is suspicious and suggests overfitting regardless. This could be due to the limited size and scope of the data however. For now we'll move on.

Looking at the confusion matrix we can see the the Adelie penguin class is most accurately predicted. 

We also train separate models on the standardised data as wel as the raw, un-scaled data also for comparions. Looking at the three RMSEs we see that the models trained on scaled data perform much better than the raw data trained model, with the standardisation method wining out albeit with a suspicously perfect RMSE of 0. This bears closer inspection.

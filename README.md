# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This data contains banking data, specifically about individuals applying for loans. The objective is to develop a model capable to predicts if
the individual will subscribe or not to the loan

The best performing model was a Voting Ensemble with an accuracy 91,4%. Many other models had similar accuracy like the best model.

## Scikit-learn Pipeline
The preprocessing pipeline performs the following steps:
- Remove NAs from the dataset.
-  One Hot Encoding of Job, Contact and Education variables.
-  Encoding other categorical variables as a number.
-  Encoding the month variable.
-  Encode the objective varible.

Once the data is prepared, a train - test split of 30% is performed, providing enough data for training the model and test the trained model in test data. After this step, a Logistic Regression classification method was used. This method uses C for regularization strength and max_iter for the maximum number of iterations.
The Azure Hyperdrive was used for hyperparameter tuning and includes the following steps:

### Parameter Sampling
I chose the Random Parameter Sampling, which doesn't require pre-specified values, this enables a broad search not limiting the space like in a grid search case.

### Early Stopping Policy
The BanditPolicy stopping policy retains models with better and/or similar performance, making it more flexible than a more rigid policy like truncation.

The best model had a 91,4% accuracy with the following parameters: C:0.408 and max iterations: 25.  

## AutoML
AutoML Pipeline follows similar steps than the Scikit-learn pipeline with the difference that is not necessary to split the data into train and test sets, the predictors and target are merged before the AutoML process and this joined dataset is used as an input for the AutoML config.
AutoML selected the best model with an accuracy of   and selecting 

## Pipeline comparison
Both models performed with a similar accuracy, the AutoML with and HyperDrive with . In case of AutoML model is an average of several individual classifiers

## Future work
In the future would be interesting to explore new feature engineer techniques, specially creating new predictors. On the other hand, with AutoML would like to experiment with longer runs.



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
I chose the Random Parameter Sampling, which doesn't require pre-specified values, but chooses combinations from the parameter space defined in it. In this particular case i defined C parameter as a value from a uniform distribution between 0 and 10, and a max_iter parameter with four discrete values.
RandomParameterSampling performs equal or better than Gridsearch sampling, but with the benefit that takes less time to complete the process

### Early Stopping Policy
The BanditPolicy stopping policy retains models with better and/or similar performance, making it more flexible than a more rigid policy like truncation. This early stopping policy stops the model to fit further if it doesn't improve anymore, this saves a lot of time.

The best model had a 91,4% accuracy with the following parameters: C:0.408 and max iterations: 25.  

## AutoML
AutoML Pipeline follows similar steps than the Scikit-learn pipeline with the difference that is not necessary to split the data into train and test sets, the predictors and target are merged before the AutoML process and this joined dataset is used as an input for the AutoML config.
AutoML selected a Voting Ensemble as the best model with an accuracy of 91.6%, with the following parameters:
- min_samples_leaf=0.01,
- min_samples_split=0.01,
- min_weight_fraction_leaf=0.0,
- n_estimators=25,
- n_jobs=1,
- oob_score=True,
- random_state=None,
- verbose=0,
- warm_start=False
 The voting method was soft voting, where all models class probabilities are averaged and the prediction is made by the highest probability. 

## Pipeline comparison
Both models performed with a similar accuracy, the AutoML with 91,9% accuracy and HyperDrive with 91,4% accuracy . In case of AutoML model is an average of several individual classifiers

## Future work
In the future would be interesting to explore new feature engineer techniques, specially creating new predictors. On the other hand, with AutoML would like to experiment with longer runs.

## Proof of cluster deletion
- Compute cluster was deleted manually.
![image](https://user-images.githubusercontent.com/47700844/116472978-9a020c00-a844-11eb-9e4f-ba35e21fba6f.png)




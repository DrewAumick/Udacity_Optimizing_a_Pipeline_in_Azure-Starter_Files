# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains information from a marketing campaign run by a bank. It has financial data about individuals who were targeted by the campaign and whether or not the campaign was successful. We seek to classify whether or not a given indivual would likely respond positively to the marketing campaign based on their finacial features.

Both the Hyperdive parameter optimization of the SKLearn Logistic Regression model and the best AutoML discovered VotingEnsemble model achieved an approximate accuracy of 91%, but the AutoML VotingEnsemble came out just slightly ahead.

## Scikit-learn Pipeline

### Pipeline Architecture
First, the data is retrieved from the provided website in the form of a CSV file. The CSV is fed into a TabularDatasetFactory, creating an Azure Tabular Dataset. This dataset is then fed into the clean_data function which converts the dataset into a pandas dataframe and converets several columns from strings to numerical values. Then the labels are seperated into a different dataframe. Finally, the data and the lables are split into training and testing sets, and the data is ready for training.

Our model for this project is an SKLearn Logistic Regression model with two hyperparameters: C, also known as the Regularization Strength, and max_iter, aka the maximum iterations for the model to converge. We use Hyperdrive to generate a random sweep of these hyperparameters and the model is trained using these parameters and the cleaned training data.

Finally, each model's accuracy is tested using the testing dataset and the model with the highest accuracy is selected. 

![Hyperdrive Run](https://github.com/DrewAumick/Udacity_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/hyperdrive%20run.PNG)

In this case, a C of about 24.8 and a max_iter of 1000 gave the best results.

### Parameter Sampling
I chose to use a RandomParameterSampling, which only samples some randomly selected values as opposed to something like a GridParameterSampling which would do all combinations of parameters. I chose this because the random sampling in general will find close to optimal solutions but requires much less computing time/resources than searching the entire grid. 

### Early Stopping Policy
I chose to use the BanditPolicy early stopping policy. The Bandit policy will stop any run where the accuracy isn't within a threshold (in my case I chose .2) of the best accuracy achieved so far. This ensures that any run that is performing badly will stop and not waste any more computing resources.

## AutoML
![AutoML Run](https://github.com/DrewAumick/Udacity_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/AutoML%20Run.PNG)

AutoML generated and attempted several different types of models, including RandomForest, XGBoost, SGD, LightGBM, StackEnsemble, and VotingEnsemble. The best it found was VotingEnsemble which did an ensemble of several of the above described models at different weights: 
```
'ensembled_algorithms': "['LightGBM', 'XGBoostClassifier', 'SGD', 'SGD', 'SGD', 'RandomForest']", 'ensemble_weights': '[0.38461538461538464, 0.23076923076923078, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.15384615384615385]'
```

## Pipeline comparison
The difference in accuracy between the two models was very minimal, less than half of a percent (91.7% vs 91.3%). But AutoML had two main advantages over the custom training script with Hyperdrive. 

The first is that AutoML tried so many different types of models and handles all the hyperparameter optimizations already. Some of the models it chose are ones I might not have even thought about trying.

The second is that by using the integrated AzureML models, you get some nice bonus features in terms of exploring the finished model that we don't get from the script. For example, the only metric returned by the script is the accuracy, so that's all we get back from those runs. But in AutoML we get a whole suite of metrics to examine from the model (see below)

![AutoML Metrics](https://github.com/DrewAumick/Udacity_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/AutoML%20Best%20Run%20Metrics.PNG)

On top of that, there's also the Model Explainability that we don't get when we write a custom script. Using the Explainations tab from our VotingEnsemble model (see below) I can see that employment variation rate and duration were both incredbily important features to this model.

![AutoML Exlpanation](https://github.com/DrewAumick/Udacity_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/AutoML%20Best%20Run%20Explanation.PNG)

## Future work
In the future, if we want to try to get Hyperdrive to compete a little more, we could try modifying our custom training script to train more types of models, perhaps looking at some of the model types that worked well in AutoML. 

We could also look at the Explanations tab from AutoML to assist with feature engineering. If there are some features that don't seem to contribute much at all, dropping them might help improve accuracy.  

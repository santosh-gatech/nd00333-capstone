# Azure ML Project - Predicting Heart Failure 

Heart failure is a common event caused by Cardiovascular diseases (CVDs) and this dataset contains 12 features that can be used to predict mortality by heart failure. 

Machine learning models can be of great importance for early detection and management for people with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease).

It's a classificaiton task and we will be using two approaches to find the solution:

1) Train a model using AutoML
2) Train a model using Hyperdrive

We will then deploy the best model as a webservice and test it by sending a request to the model endpoint.

## Project Set Up and Installation

This project has been implemented in Azure ML.

1) First you need to setup a workplace in Azure ML Studio.

2) Then create a compute instance VM to run jupyter notebooks.

3) You can then install some starter files for following some steps for running an experiment using AutoML and Hyperdrive:
https://github.com/udacity/nd00333-capstone/tree/master/starter_file

4) Then you can proceed to select the dataset of your own choice. Here, following dataset was chosen from Kaggle:
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

5) Select the best performing model (AutoML or Hyperdrive) and deploy it.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

# Azure ML Project - Predicting Heart Failure 

This project demonstrates the ability to use an external dataset in Azure ML workspace, train a model using the different tools available in the AzureML framework as well as the ability to deploy the model as a web service.

We will be using two approaches to find the solution:

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

I will be using Heart Failure Prediction dataset from Kaggle:
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

Heart failure is a common event caused by Cardiovascular diseases (CVDs) and this dataset contains 12 features that can be used to predict mortality by heart failure. 

Machine learning models can be of great importance for early detection and management for people with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease).

### Task

It's a classificaiton task. We will be predicting a death event by heart failure when given 12 features as an input.

In total, the dataset has 299 samples.

Following are the 12 features we will be using as an input for our ML models:

1) age
2) anaemia - Decrease of red blood cells or hemoglobin (boolean)
3) creatinine_phosphokinase - Level of the CPK enzyme in the blood (mcg/L)
4) diabetes - If the patient has diabetes (boolean)
5) ejection_fraction - Percentage of blood leaving the heart at each contraction (percentage)
6) high_blood_pressure - If the patient has hypertension (boolean)
7) platelets - Platelets in the blood (kiloplatelets/mL)
8) serum_creatinine - Level of serum creatinine in the blood (mg/dL)
9) serum_sodium - Level of serum sodium in the blood (mEq/L)
10) sex - Woman or man (binary)
11) smoking - If the patient smokes or not (boolean)
12) time - Follow-up period (days)

Our task will be to predict Death Event - If the patient deceased during the follow-up period (boolean)

### Access

First, I downloaded the dataset from Kaggle. Then I uploaded it on my github project repo to access it.

Following is the code sample of how I accessed the data in my workspace.

```
key = "Heart-failure"
description_text = "Heart Failure Prediction DataSet from Kaggle"

# Create AML Dataset and register it into Workspace
example_data = 'https://raw.githubusercontent.com/santosh-gatech/nd00333-capstone/master/starter_file/heart_failure_clinical_records_dataset.csv'
dataset = Dataset.Tabular.from_delimited_files(example_data)        

# Register Dataset in Workspace
dataset = dataset.register(workspace=ws,
                           name=key,
                           description=description_text)
```

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

# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

 During the evaluation phase, XGboost and Logistic Regression and Random Forest were considered, with XGBoost selected due to its better performance.

## Intended Use

This predictive model is designed to determine whether an individual's annual income surpasses $50,000.

## Training Data

UCI Census Income Dataset
For categorical data, missing values were addressed by imputing using SimpleImputer, utilizing the most frequent value. Categories were encoded using LabelEncoder.

## Evaluation Data

The training data is divided using the `train_test_split` function from `sklearn`, implementing stratification based on the salary.

## Metrics

Evaluation metrics includes Precision, Recall and F1 score.

## Ethical Considerations

Data is open sourced on UCI for educational purposes.

## Caveats and Recommendations

The data was collected in 1996 which does not reflect insights from the modern world.
Features with minor categories should be focused more when collecting extra data.

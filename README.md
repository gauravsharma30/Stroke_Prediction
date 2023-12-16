# Stroke_Prediction

The goal of this model is to figure out if person will experience a stroke on the basis of their age, nature of work, urban/rural resident, marital status, and several clinical levels.

## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

Files
stroke_train_set.csv - the training set with input features and GT value for 'stroke'
stroke_test_set_nogt.csv - the test set with only input features to evaluate your model
sample_submission.csv - a sample submission file in the expected format
[data](https://github.com/gauravsharma30/Stroke_Prediction/tree/main/Data)

- gender: "Male", "Female" or "Other"

- age: age of the patient

- hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension

- heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease

- ever_married: "No" or "Yes"

- work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"

- Residence_type: "Rural" or "Urban"

- avg_glucose_level: average glucose level in blood

- bmi: body mass index

- smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*

- stroke: 1 if the patient had a stroke or 0 if not (Prediction)


# Machine Learning Model used:

I have used the following machine learning models in this project:

## 1. K-Nearest Neighbors (KNN)
K-Nearest Neighbors is a simple yet powerful classification algorithm. It classifies a data point based on the majority class of its k-nearest neighbors in the feature space. [Learn more](https://scikit-learn.org/stable/modules/neighbors.html#classification)

## 2. Support Vector Classifier (SVC)
Support Vector Classifier, a type of Support Vector Machine (SVM), is a versatile algorithm for both classification and regression tasks. It works by finding the optimal hyperplane that best separates classes in the feature space. [Learn more](https://scikit-learn.org/stable/modules/svm.html#classification)

## 3. Random Forest
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) of the individual trees. [Learn more](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)

## 4. AdaBoost
AdaBoost is a boosting algorithm that combines weak classifiers to form a strong classifier. It sequentially corrects the errors of the weak classifiers and improves overall performance. [Learn more](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

## 5. Multi-Layer Perceptron (MLP)
Multi-Layer Perceptron, a type of artificial neural network, consists of multiple layers of nodes (neurons) and is capable of learning complex patterns. It is widely used for both classification and regression tasks. [Learn more](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

Feel free to explore the provided links for a deeper understanding of each algorithm's functionality and parameters in scikit-learn.


## Preprocessing Steps

### 1. Handle Duplicates and Missing Values

- Fill missing values in the 'bmi' column with the mean value.

### 2. One-Hot Encoding for Categorical Columns

- Utilize one-hot encoding for categorical columns such as 'gender', 'ever_married', 'work_type', 'Residence_type', and 'smoking_status'.
- Resulting in the creation of binary columns for each category.

### 3. Feature Selection

#### Feature Selection Methods:

1. **Lasso Regression:**
   - **Explanation:** Lasso Regression is a linear regression technique that adds a penalty term to the absolute values of the coefficients. It encourages sparsity in the model, making some coefficients exactly zero.
   - **Link:** [Lasso Regression - Scikit-Learn](https://scikit-learn.org/stable/modules/linear_model.html#lasso)

2. **Recursive Feature Elimination (RFE):**
   - **Explanation:** RFE is a feature selection method that recursively removes the least important features until the desired number of features is achieved. It uses model accuracy as a criterion for feature ranking.
   - **Link:** [Recursive Feature Elimination - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)

3. **Mutual Information Classification:**
   - **Explanation:** Mutual Information is a measure of the dependency between two variables. In the context of classification, Mutual Information Classification quantifies the relationship between features and the target variable.
   - **Link:** [Mutual Information - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)

4. **Chi-squared (Chi2) Test:**
   - **Explanation:** The chi-squared test is used to determine the association between categorical features and the target variable. It measures how expected frequencies of observations differ from actual frequencies.
   - **Link:** [Chi-squared Test - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)

### 4. Address Class Imbalance

#### Over-Sampling

- Apply over-sampling techniques to handle class imbalance.

#### Under-Sampling

- Implement under-sampling techniques for a balanced dataset.

## Model Training and Evaluation

- Train various machine learning models using the preprocessed data.
- Evaluate each model's performance using metrics such as accuracy, F1 score, and confusion matrix.


# Some Visualizations:

- [Scatter plot matrix](https://github.com/gauravsharma30/Stroke_Prediction/blob/main/Pictures/scatter_plot.png)

- [Correlation](https://github.com/gauravsharma30/Stroke_Prediction/blob/main/Pictures/correlation.png)




##Metric used:

## F1 Score

**F1 Score** is a metric that combines precision and recall into a single value. It is the harmonic mean of precision and recall and provides a balance between the two. F1 Score is particularly useful when there is an uneven class distribution.

- [Learn more about F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

## Accuracy

**Accuracy** is a common metric for evaluating classification models. It measures the ratio of correctly predicted instances to the total instances. While accuracy is widely used, it may not be suitable for imbalanced datasets, as it doesn't account for the distribution of classes.

- [Learn more about Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

# Evaluation Score


## KNN
|                   | Normal    | Over-Sampling | Feature Multiinfo | Feature Chi2 | Feature RFE | Hyperparameter |
|-------------------|-----------|---------------|-------------------|--------------|-------------|----------------|
| **F1 Score 0**    |   0.97    |    0.92       |      0.91         |    0.92      |   0.93      |     0.95       |
| **F1 Score 1**    |   0.00    |    0.12       |      0.19         |    0.16      |   0.11      |     0.22       |
| **Accuracy**      |   0.94    |    0.81       |      0.84         |    0.85      |   0.87      |     0.90       |


## SVC
|                   | Normal    | Over-Sampling | Feature Multiinfo | Feature Chi2 | Feature RFE | Hyperparameter |
|-------------------|-----------|---------------|-------------------|--------------|-------------|----------------|
| **F1 Score 0**    |   0.87    |    0.88       |      0.82         |    0.83      |   0.81      |     0.91       |
| **F1 Score 1**    |   0.21    |    0.22       |      0.19         |    0.20      |   0.20      |     0.21       |
| **Accuracy**      |   0.71    |    0.79       |      0.70         |    0.72      |   0.70      |     0.84       |

## Random Forest
|                   | Normal    | Over-Sampling | Feature Multiinfo | Feature Chi2 | Feature RFE | Hyperparameter |
|-------------------|-----------|---------------|-------------------|--------------|-------------|----------------|
| **F1 Score 0**    |   0.98    |    0.97       |      0.96         |    0.96      |   0.96      |     0.96       |
| **F1 Score 1**    |   0.05    |    0.08       |      0.14         |    0.18      |   0.08      |     0.16       |
| **Accuracy**      |   0.95    |    0.94       |      0.92         |    0.92      |   0.92      |     0.92       |


## AdaBoost
|                   | Normal    | Over-Sampling | Feature Multiinfo | Feature Chi2 | Feature RFE | Hyperparameter |
|-------------------|-----------|---------------|-------------------|--------------|-------------|----------------|
| **F1 Score 0**    |   0.97    |    0.92       |      0.89         |    0.90      |   0.90      |     0.91       |
| **F1 Score 1**    |   0.00    |    0.12       |      0.21         |    0.23      |   0.23      |     0.24       |
| **Accuracy**      |   0.95    |    0.81       |      0.81         |    0.83      |   0.83      |     0.83       |

## MLP
|                   | Normal    | Over-Sampling | Feature Multiinfo | Feature Chi2 | Feature RFE | Hyperparameter |
|-------------------|-----------|---------------|-------------------|--------------|-------------|----------------|
| **F1 Score 0**    |   0.97    |    0.92       |      0.91         |    0.90      |   0.88      |     0.90       |
| **F1 Score 1**    |   0.04    |    0.12       |      0.22         |    0.26      |   0.22      |     0.26       |
| **Accuracy**      |   0.95    |    0.81       |      0.83         |    0.86      |   0.84      |     0.82       |



## Key Takeaways

The challenging thing was to address classs imbalance, and how to match the columns after feature selection with the output code to produce result.
- SVC worked best for the classification with this data, although I thought MLP would produced better result but it didn't happen.
- SVC with over sampling gave better result, comapared to using both over and under sampling.
- Hyperparameter tuning for MLp model takes lots of computaion ppower, so not effective for machine with low hardware capacity.

## How to Run

The code is built on Google Colab on an iPython Notebook. 

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

What are the future modification you plan on making to this project?

- Try more models

- Try to look for better feature selection and hyperparameter tuninig which doesn't use much computational power.


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at gauarvsharma.dsml@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


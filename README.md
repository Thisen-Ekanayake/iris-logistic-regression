# Logistic Regression on Iris Dataset
This project demonstrates the application of logistic regression to classify the Iris dataset. It includes data preprocessing, hyperparameter tuning, model evaluation, and visualization. The goal is to build an optimized logistic regression model and analyze feature importance.

## Features
- **Data Loading & Preprocessing:** Loads the Iris dataset, splits it into training and testing sets, and scales the features for better performance.
- **Hyperparameter Tuning:** Uses GridSearchCV to find the optimal hyperparameters for the logistic regression model.
- **Model Evaluation:** Evaluates the model using accuracy and a confusion matrix.
- **Feature Importance Analysis:** Visualizes the impact of features on the target classification using coefficients.
- **Visualizations:** Includes pair plots and heatmaps to explore feature relationships.
- **Model Saving:** Saves the trained model for future use.

## Prerequisites
#### To run this project, ensure you have the following:
- Python 3.8+

#### Libraries:
- joblib
- pandas
- matplotlib
- seaborn
- scikit-learn

## How to Use
#### 1. Clone this repository or download the files.
#### 2. Place the *iris.csv* file in the same directory as the script.
#### 3. Install the required libraries:
```
pip install joblib pandas matplotlib seaborn scikit-learn
```

#### 4. Run the script:
```
python iris.py 
```
#### 5. View the visualizations and outputs in the console or as plots.


## Outputs
#### Model Performance:
- Confusion Matrix
- Test Accuracy
#### Visualizations:
- Pair plots showing feature relationships
- Correlation heatmap
- Bar plot for feature importance (coefficients)
#### Saved Model:
- tuned_iris_model.pkl: The trained logistic regression model with optimal hyperparameters.

## Notes
- The Iris dataset must be in the same directory as the script.
- Ensure the dataset format matches the expected structure.

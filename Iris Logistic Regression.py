# Importing relevant libraries
import joblib  # For saving and loading models
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For visualizations
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # For model evaluation
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV  # For model training, testing, and hyperparameter tuning
from sklearn.preprocessing import StandardScaler  # For feature scaling

"""
Project: Logistic Regression on Iris Dataset with Hyperparameter Tuning
Author: Thisen Ekanayake
Description:
    This script trains a logistic regression model on the Iris dataset.
    It includes data preprocessing, hyperparameter tuning using GridSearchCV,
    model evaluation, and feature importance analysis. The trained model is
    saved for future use.

Features:
    - Data loading and preprocessing
    - Hyperparameter tuning using GridSearchCV
    - Model evaluation with accuracy and confusion matrix
    - Visualization of data relationships and feature importance

How to Use:
    1. Place the 'iris.data' file in the same directory as this script.
    2. Install the required libraries (joblib, pandas, matplotlib, seaborn, scikit-learn).
    3. Run the script to train and save the model.
    4. Visualize outputs like confusion matrix and feature importance plots.

Output:
    - Tuned model saved as 'tuned_iris_model.pkl'
    - Visualizations of feature relationships and model performance.
"""

# Loading the dataset (Iris dataset)
data = pd.read_csv("iris.data", header=None)
# Naming the columns for better understanding of the data
data.columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"]

# Splitting the features (X) and target (y)
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data["Species"]  # Target (Species column)

# Splitting the data into training (70%) and testing (30%) sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling to standardize the features (important for gradient-based algorithms like logistic regression)
scaler = StandardScaler()
XtrainScaled = scaler.fit_transform(Xtrain)  # Fit and transform on the training data
XtestScaled = scaler.transform(Xtest)  # Only transform on the test data to prevent data leakage

# Defining the hyperparameter grid for tuning the logistic regression model
paramGrid = {
    "C" : [0.01, 0.1, 1, 10, 50, 100],  # Regularization strength
    "penalty" : ["l1", "l2", "elasticnet"],  # Regularization type
    "solver" : ["saga"],  # Solver for optimization
    "max_iter" : [500, 1000, 2000],  # Number of iterations for optimization
    "l1_ratio" : [0.1, 0.5, 0.9]  # Elastic net mixing parameter (only used when penalty="elasticnet")
}

# Setting up GridSearchCV for hyperparameter tuning
gridSearch = GridSearchCV(estimator=LogisticRegression(verbose=1), param_grid=paramGrid, scoring='accuracy', cv=5, n_jobs=-1, error_score="raise")
gridSearch.fit(XtrainScaled, ytrain)  # Perform the grid search on the scaled training data

# Getting the best model and its parameters after grid search
bestModel = gridSearch.best_estimator_
print(f"Best Parameters: {gridSearch.best_params_}")
print(f"Best Cross-Validation Accuracy: {gridSearch.best_score_ * 100:.2f}%")

# Predicting the target for the test data using the best model
ypred = bestModel.predict(XtestScaled)

# Evaluating the model using accuracy score
accuracy = accuracy_score(ytest, ypred)
print(f"Test Accuracy of Tuned Model: {accuracy * 100:.2f}%")

# Plotting the confusion matrix to evaluate the model's performance
cm = confusion_matrix(ytest, bestModel.predict(XtestScaled))  # Generating the confusion matrix
ConfusionMatrixDisplay(cm, display_labels=bestModel.classes_).plot(cmap="Blues")  # Displaying the confusion matrix
plt.title("Confusion Matrix")  # Adding a title to the plot
plt.show()

# Creating a pairplot to visualize the relationships between features, color-coded by species
sns.pairplot(data, hue="Species", diag_kind="kde", palette="Set2")  # Pairplot with KDE on the diagonal
plt.show()

# Creating a correlation heatmap to show the relationships between features
correlationMatrix = data.iloc[:,:-1].corr()  # Compute correlation matrix (excluding the target column)
plt.figure(figsize=(8, 6))  # Setting the size of the plot
sns.heatmap(correlationMatrix, annot=True, cmap="coolwarm", fmt=".2f")  # Creating the heatmap
plt.title("Feature Correlation Heatmap")  # Adding a title
plt.show()

# Extract feature importance (coefficients) from the best model
coefficients = bestModel.coef_[0]  # Coefficients for the first class
features = X.columns  # Feature names

# Create a DataFrame to store feature importance (without abs)
importance_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
})

# Sort by coefficient values (keeping sign to reflect direction)
importance_df = importance_df.sort_values(by="Coefficient", ascending=False)

# Plot the feature importance with original coefficients (considering both positive and negative effects)
plt.figure(figsize=(8, 6))
sns.barplot(
    x="Feature", y="Coefficient", data=importance_df,
    palette="coolwarm"  # Color scheme reflecting positive/negative values
)
plt.title("Feature Importance with Original Coefficients (Logistic Regression)", fontsize=14)
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()

# Print the feature importance with coefficients
print(importance_df)

# Saving the trained model to a file using joblib
joblib.dump(bestModel, "tuned_iris_model.pkl")  # Save the best model to a file
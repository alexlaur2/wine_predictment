import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib
import numpy as np

# Load dataset
df = pd.read_excel('B9_winequality-white.xlsx')

# Convert non-numeric values to NaN and fill NaN with column mean
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())


# Outlier treatment using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


for col in df.columns:
    if col != 'quality':
        df = remove_outliers(df, col)

# Create binary target variable 'best quality'
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

# Separate features and target
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(features_scaled, target, test_size=0.2, random_state=40)

# Define the models to be evaluated
models = [
    LogisticRegression(),
    SVC(kernel='rbf'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier(),
    KNeighborsClassifier()
]

# Evaluate each model with hyperparameter tuning
best_model = None
best_score = 0
for model in models:
    if isinstance(model, (LogisticRegression, SVC, KNeighborsClassifier)):
        param_grid = {'C': [0.1, 1, 10]} if isinstance(model, (LogisticRegression, SVC)) else {
            'n_neighbors': [5, 10, 15]}
    elif isinstance(model, DecisionTreeClassifier):
        param_grid = {'max_depth': [None, 10, 20, 30]}
    elif isinstance(model, RandomForestClassifier):
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    elif isinstance(model, GradientBoostingClassifier):
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    elif isinstance(model, XGBClassifier):
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(xtrain, ytrain)

    best_model_candidate = grid_search.best_estimator_
    train_auc = metrics.roc_auc_score(ytrain, best_model_candidate.predict(xtrain))
    test_auc = metrics.roc_auc_score(ytest, best_model_candidate.predict(xtest))

    print(f"{best_model_candidate.__class__.__name__}:")
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Validation AUC: {test_auc:.4f}\n")

    if test_auc > best_score:
        best_score = test_auc
        best_model = best_model_candidate

print(f"Best model: {best_model.__class__.__name__} with Validation AUC: {best_score:.4f}")
joblib.dump(best_model, 'best_model.pkl')

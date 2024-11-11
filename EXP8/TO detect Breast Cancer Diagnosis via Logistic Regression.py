import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
df = pd.read_csv('data.csv')
df.drop(columns=["id", "Unnamed: 32"], inplace=True)
df


X = df.iloc[:, 1:]
y = df.iloc[:, 0]
y = np.where(y == 'B', 0, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1, 10],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


best_log_reg = grid_search.best_estimator_
preds_optimized = best_log_reg.predict(X_test)


accuracy_optimized = best_log_reg.score(X_test, y_test)
print("Accuracy of Optimized Logistic Regression:", accuracy_optimized)


print("Classification Report for Optimized Logistic Regression:\n", classification_report(y_test, preds_optimized))

# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import pandas as pd

iris_data = load_iris()

iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target

iris_df_setosa = iris_df.copy()
iris_df_setosa['target'] = (iris_df['target'] == 0).astype(int)  # Class Setosa vs rest

iris_df_versicolor = iris_df.copy()
iris_df_versicolor['target'] = (iris_df['target'] == 1).astype(int)  # Class Versicolor vs rest

iris_df_virginica = iris_df.copy()
iris_df_virginica['target'] = (iris_df['target'] == 2).astype(int)  # Class Virginica vs rest

trained_classifiers = []

# Training and evaluating logistic regression classifiers for each binary classification task
for index, data in enumerate([iris_df_setosa, iris_df_versicolor, iris_df_virginica]):
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    print(f"Training and evaluating classifier for class: {class_names[index]}")
    
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :4], data.iloc[:, -1], test_size=0.2, random_state=42)
    
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    predictions = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(classification_report(y_test, predictions))
    
    trained_classifiers.append(classifier)

X_train, X_test, y_train, y_test = train_test_split(iris_df.iloc[:, :4], iris_df.iloc[:, -1], test_size=0.2, random_state=42)

individual_predictions = [classifier.predict(X_test) for classifier in trained_classifiers]
# Combining predictions using majority voting
final_predictions = [max(range(len(trained_classifiers)), key=lambda i: preds[i]) for preds in zip(*individual_predictions)]

accuracy = accuracy_score(y_test, final_predictions)
print(classification_report(y_test, final_predictions))
print(f"Accuracy for Logistic Regression Ensemble: {accuracy}")

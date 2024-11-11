import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load train and test data
# train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_train_df = train_df.sample(frac=0.01, random_state=42)
sample_train_df.shape

X = sample_train_df.drop('label', axis=1)
y = sample_train_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.decomposition import PCA
pca = PCA(n_components=115)  
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


selected_features = pca.components_

print("Selected Features by PCA:")
for i, component in enumerate(selected_features):
    print(f"Principal Component {i+1}:")
    for j, feature in enumerate(X.columns):
        print(f"Feature {feature}: {component[j]}")
    print()



svm_model = SVC()

svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5)
svm_grid_search.fit(X_train_pca, y_train)

best_svm_params = svm_grid_search.best_params_
print("Best Parameters for SVM after PCA:", best_svm_params)

best_svm_model = svm_grid_search.best_estimator_

y_svm_pred = best_svm_model.predict(X_test_pca)

svm_accuracy = accuracy_score(y_test, y_svm_pred)
print("SVM Accuracy after PCA:", svm_accuracy * 100)


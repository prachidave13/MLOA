# %% [markdown]
# To Implementing Predicting Energy Efficiency for Residential Buildings

# %%

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('ENB2012_data.xlsx')

df  

# %%
corr = df.corr()
X,y = df.iloc[:,:-2].values.T, df.iloc[:,-2:].values.T

# %%
d, N = X.shape
np.random.seed(2)
r = np.random.permutation(N) 
Xtr = X[:, r[128:]]
Ytr = y[:, r[128:]]
Xtel = X[:, r[:128]]
Ytel = y[:, r[:128]]

ym1 = np.mean(Ytel, axis=1) 
ind1 = np.argsort(ym1)
Xte = Xtel[:, ind1] 
Yte = Ytel[:, ind1]


Xtr_tilde = np.vstack([Xtr, np.ones((1, Xtr.shape[1]))])
 

W = np.linalg.lstsq(Xtr_tilde.T, Ytr.T, rcond=None)[0]

Xte_tilde = np.vstack([Xte, np.ones((1, Xte.shape[1]))]) 
Yp = np.dot(W.T, Xte_tilde)

error = np.linalg.norm(Yte - Yp, 'fro') / np.linalg.norm(Yte, 'fro')

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(Yte[0, :], 'b', linewidth=2)
plt.plot(Yp[0, :],'r--', linewidth=2) 
plt.title('Heating Load Prediction')
plt.legend(['True Heating Load', 'Predicted Heating Load'])
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 2)
plt.plot(Yte[1, :], 'g', linewidth=2)
plt.plot(Yp[1, :],'m--', linewidth=2) 
plt.title('Cooling Load Prediction')
plt.legend(['True Cooling Load', 'Predicted Cooling Load'])

# %%
X_train,X_test,y_train,y_test = train_test_split(X.T,y.T,random_state = 42, test_size = 0.2)
 
regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X_train, y_train) 
linear = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
lasso = MultiOutputRegressor(Lasso(alpha=4)).fit(X_train, y_train) 
models = [regr, linear, lasso]

# %%
preds = [] 
names = { 1: "Ridge",
2: "Linear Regression",
3: "Lasso"
}
for i,model in enumerate(models):
    preds = model.predict(X_test)
    print(f"The MSE for {names[i+1]} is: {mean_squared_error(y_test,preds)}") 
    print(f"The R2 score for {names[i+1]} is: {r2_score(y_test,preds)}")
    print("-"*20)
    print()

y_pred = linear.predict(X_test)

# %%
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(y_test[0, :], 'b', linewidth=2) 
plt.plot(y_pred[0, :],'r--', linewidth=2) 
plt.title('Heating Load Prediction')
plt.legend(['True Heating Load', 'Predicted Heating Load']) 
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 2)
plt.plot(y_test[1, :], 'g', linewidth=2) 
plt.plot(y_pred[1, :],'m--', linewidth=2) 
plt.title('Cooling Load Prediction')
 
plt.legend(['True Cooling Load', 'Predicted Cooling Load'])

# %%

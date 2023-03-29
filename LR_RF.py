########Lets write some code baby#############

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Reads and loads dataset in pandas DataFrame
df = pd.read_csv("Dilute_Solute_Diffusion_with_features.csv")
# df.head()
# df.describe()

###Creating output (y) and input (X) datasets
y = df["E_raw (eV)"].values

excluded = ["Material compositions 1", "Material compositions 2", "Enorm (eV)", "E_raw (eV)"]

X = df.drop(excluded, axis=1)
# X.head()



####Appy linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict 

lr = LinearRegression()

lr.fit(X, y)

# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))


# Use 10-fold cross validation (90% training, 10% test)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)


print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))

#plotting results (predicted vs expected)
goal = np.linspace(1, 6.2, 10)

plt.figure(dpi=100)
plt.scatter(y,cross_val_predict(lr, X, y, cv=crossvalidation))
plt.plot(goal, goal, 'r--')
plt.xlabel("DFT (MP) Diffusion Energy Barrier (eV)");
plt.ylabel("Predicted Diffusion Energy Barrier (eV)");



####Appy random forest model
from sklearn.ensemble import RandomForestRegressor


rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(X, y)
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))


# compute cross validation scores for random forest model
r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


plt.figure(dpi=100)
plt.scatter(y,cross_val_predict(lr, X, y, cv=crossvalidation))
plt.plot(goal[:350], goal[:350], 'r--')
plt.xlabel("DFT (MP) Diffusion Energy Barrier (eV))");
plt.ylabel("Random forest Diffusion Energy Barrier (eV)");

##plotting results
plt.figure(dpi=100)
importances = rf.feature_importances_
included = X.columns.values
indices = np.argsort(importances)[::-1]
plt.bar(included[indices][0:10], importances[indices][0:10])
plt.xticks(rotation=30, ha='right')
plt.show()

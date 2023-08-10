


!pip install shap

import xgboost
import shap
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

/content/drive/MyDrive/Boston.csv

import pandas as pd
raw_df = pd.read_csv("/content/drive/MyDrive/Boston.csv")
raw_df.head(20)

raw_df1=raw_df.drop(['Unnamed: 15'], axis=1)

raw_df2=raw_df1.drop(['Unnamed: 16'], axis=1)

data = np.hstack([raw_df2.values[::2, :], raw_df2.values[1::2, :2]])
target = raw_df2.values[1::2, 2]



data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header= "none")
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

raw_df2.head()

X = raw_df2.drop("MEDV", axis=1)
y = raw_df2["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X=data
y=target


model = xgboost.XGBRegressor().fit(X, y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred = xgb_model.predict(X_test)

# Calculate the accuracy of the model
accuracy_xgb = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy_xgb)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Calculate the accuracy of the model
y_pred = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy_rf)



# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# summarize the effects of all the features
shap.plots.beeswarm(shap_values)

shap.plots.bar(shap_values, max_display=15)

shap.plots.bar(shap_values[0], max_display=15)

rf = RandomForestRegressor(n_estimators=150)
model2 = rf.fit(X_train, y_train)



sort = rf.feature_importances_.argsort()
plt.barh(raw_df2.columns[sort], rf.feature_importances_[sort])
plt.xlabel("Feature Importance")

xg = GradientBoostingRegressor(n_estimators=150)
modelxgb = xg.fit(X_train, y_train)

sort = xg.feature_importances_.argsort()
plt.barh(X.columns[sort], xg.feature_importances_[sort])
plt.xlabel("Feature Importance")

explainer2 = shap.Explainer(model2)
shap_values2 = explainer2(X)

shap.plots.bar(shap_values2, max_display=15)

shap.plots.bar(shap_values2[0], max_display=15)










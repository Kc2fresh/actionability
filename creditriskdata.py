from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import xgboost

!pip install shap

df_gcredit = pd.read_csv("/content/drive/MyDrive/g_credit_data_risk.csv",na_values=['?'])
df_gcredit.head()

df_gcredit.head()

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap


df_gcredit = pd.get_dummies(df_gcredit, prefix=['Risk'], columns=['Risk'])




df_gcredit.head(5)

df_gcredit = df_gcredit.drop(["Risk_bad"], axis=1)

column_headers = list(df_gcredit.columns.values)
print("The Column Header :", column_headers)

df_gcredit = df_gcredit.drop(['Unnamed: 0'], axis=1)

df_gcredit = pd.get_dummies(df_gcredit)

X = df_gcredit.iloc[:,:-1]
y = df_gcredit.iloc[:,-1]

model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

shap.plots.beeswarm(shap_values)

shap.plots.bar(shap_values, max_display=26)

shap.plots.bar(shap_values[0], max_display=26)

shap.plots.bar(shap_values.abs[1], max_display=15)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=150)
model2 = rf.fit(X_train, y_train)

sort = rf.feature_importances_.argsort()
plt.barh(X.columns[sort], rf.feature_importances_[sort])
plt.xlabel("Feature Importance")

explainer2 = shap.Explainer(model2)
shap_values2 = explainer2(X)

shap.plots.bar(shap_values2, max_display=15)

shap.plots.bar(shap_values2[0], max_display=15)

shap.plots.bar(shap_values2.abs[1], max_display=15)

shap.plots.bar(shap_values2[1], max_display=15)

shap.plots.bar(shap_values2[0], max_display=15)


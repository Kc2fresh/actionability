from google.colab import drive
drive.mount('/content/drive')

!pip install shap

import xgboost
import shap
import pandas as pd


X, y = shap.datasets.diabetes()
model = xgboost.XGBRegressor().fit(X, y)

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/diabetes.csv")
df.head()

X=df.drop(["Outcome"], axis=1)

y=df["Outcome"]

X.head()

y.head()

model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)


# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# summarize the effects of all the features
shap.plots.beeswarm(shap_values)

shap.plots.bar(shap_values)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=150)
model2 = rf.fit(X_train, y_train)

sort = rf.feature_importances_.argsort()
plt.barh(X.columns[sort], rf.feature_importances_[sort])
plt.xlabel("Feature Importance")

xg = GradientBoostingRegressor(n_estimators=150)
modelxgb = xg.fit(X_train, y_train)

sort = xg.feature_importances_.argsort()
plt.barh(X.columns[sort], xg.feature_importances_[sort])
plt.xlabel("Feature Importance")

explainer2 = shap.Explainer(model2)
shap_values2 = explainer2(X)

shap.plots.bar(shap_values2, max_display=15)




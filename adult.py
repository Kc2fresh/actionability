!pip install shap


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import xgboost
import shap

X, y = shap.datasets.adult()
#model = xgboost.XGBRegressor().fit(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

print(X.shape)

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

 #Evaluate the F1 score metrics for each model
rf_preds = rf_model.predict(X_test)
rf_f1 = f1_score(y_test, rf_preds)

xgb_preds = xgb_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_preds)

print(f"Random forest F1 score: {rf_f1}")
print(f"XGBoost F1 score: {xgb_f1}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Read the CSV file using pandas
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data.drop('target_column', axis=1),  # Features
    data['target_column'],  # Target variable
    test_size=0.2,  # 20% of data for testing
    random_state=42  # Set a random seed for reproducibility
)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_data, train_labels)

# Train an xgboost classifier
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(train_data, train_labels)

# Evaluate the F1 score metrics for each model
rf_preds = rf.predict(test_data)
rf_f1 = f1_score(test_labels, rf_preds)

xgb_preds = xgb.predict(test_data)
xgb_f1 = f1_score(test_labels, xgb_preds)

print(f"Random forest F1 score: {rf_f1}")
print(f"XGBoost F1 score: {xgb_f1}")












dir(xgboost)

accuracy_rf = model.accuracy_score(X, y)
print("Random Forest Accuracy:", accuracy_rf)

column_headers = list(X.columns.values)
print("The Column Header :", column_headers)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0])

# summarize the effects of all the features
shap.plots.beeswarm(shap_values)

shap.plots.bar(shap_values,max_display=26)

shap.plots.bar(shap_values[0],max_display=26)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=150)
model2 = rf.fit(X_train, y_train)

accuracy_rf = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy_rf)

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

shap.plots.bar(shap_values2[0], max_display=15)

column_headers = list(X.columns.values)
print("The Column Header :", column_headers)

perm_importance = permutation_importance(rf, X_test, y_test)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")

#!/usr/bin/env python
# coding: utf-8

# In[19]:


# ============================================================
# MASTER SETUP - AGILE PROJECT DELAY RISK RESEARCH
# Run this cell after every kernel restart
# ============================================================

# -------------------------
# 1. IMPORT LIBRARIES
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from xgboost import XGBClassifier


# -------------------------
# 2. LOAD DATASET
# -------------------------

df = pd.read_csv("project_risk_raw_dataset.csv")

print("Dataset loaded:", df.shape)


# -------------------------
# 3. CREATE BINARY TARGET
# -------------------------

# Low / Medium = 0
# High / Critical = 1

df["Elevated_Risk"] = (
    df["Risk_Level"]
    .isin(["High", "Critical"])
    .astype(int)
)


# -------------------------
# 4. CREATE AGILE / IT SUBSET
# -------------------------

agile_it_df = df[
    (df["Project_Type"] == "IT") &
    (df["Methodology_Used"].isin(["Agile", "Scrum", "Kanban"]))
].copy()

print("Agile/IT subset:", agile_it_df.shape)

print("\nElevated risk distribution:")
print(agile_it_df["Elevated_Risk"].value_counts())


# -------------------------
# 5. CREATE X AND y
# -------------------------

y = agile_it_df["Elevated_Risk"]

X = agile_it_df.drop(
    columns=[
        "Risk_Level",
        "Elevated_Risk",
        "Project_ID"
    ]
)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# -------------------------
# 6. IDENTIFY FEATURE TYPES
# -------------------------

numeric_features = [
    col for col in X.columns
    if is_numeric_dtype(X[col])
]

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# -------------------------
# 7. PREPROCESSING
# -------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -------------------------
# 8. TRAIN / TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -------------------------
# 9. LOGISTIC REGRESSION
# -------------------------

log_reg_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# -------------------------
# 10. RANDOM FOREST
# -------------------------

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# -------------------------
# 11. XGBOOST
# -------------------------

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)


# -------------------------
# 12. DUMMY BASELINE
# -------------------------

dummy_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]
)


# -------------------------
# 13. CROSS-VALIDATION SETUP
# -------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# -------------------------
# 14. CONFIRM SETUP
# -------------------------

print("\n--------------------------------")
print("MASTER SETUP COMPLETE")
print("--------------------------------")
print("Dataset:", df.shape)
print("Agile/IT subset:", agile_it_df.shape)
print("Training:", X_train.shape)
print("Testing:", X_test.shape)
print("\nModels ready:")
print("- Dummy Classifier")
print("- Logistic Regression")
print("- Random Forest")
print("- XGBoost")


# In[20]:


# Train baseline models

dummy_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

print("All baseline models trained successfully.")


# In[21]:


print("df:", df.shape)
print("agile_it_df:", agile_it_df.shape)
print("X:", X.shape)
print("y:", y.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nAll required variables are now loaded.")


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("Libraries imported successfully.")


# In[2]:


df = pd.read_csv("project_risk_raw_dataset.csv")

print("Dataset loaded successfully.")
print("Rows and columns:", df.shape)

df.head()


# In[3]:


print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# In[4]:


missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print("Total missing values:", df.isnull().sum().sum())
print("\nColumns with missing values:")
print(missing_values)

print("\nDuplicate rows:", df.duplicated().sum())


# In[5]:


df["Risk_Level"].value_counts()


# In[6]:


df["Risk_Level"].value_counts().plot(kind="bar")
plt.title("Distribution of Risk Levels")
plt.xlabel("Risk Level")
plt.ylabel("Number of Projects")
plt.show()


# In[7]:


df["Elevated_Risk"] = df["Risk_Level"].isin(["High", "Critical"]).astype(int)

df[["Risk_Level", "Elevated_Risk"]].head(10)


# In[8]:


df["Elevated_Risk"].value_counts()


# In[9]:


df["Elevated_Risk"].value_counts().plot(kind="bar")
plt.title("Binary Target Distribution: Elevated Risk vs Non-Elevated Risk")
plt.xlabel("Elevated Risk: 0 = Low/Medium, 1 = High/Critical")
plt.ylabel("Number of Projects")
plt.show()


# In[10]:


print("Project types:")
print(df["Project_Type"].value_counts())

print("\nMethodologies:")
print(df["Methodology_Used"].value_counts())


# In[11]:


df["Project_Type"].value_counts().plot(kind="bar")
plt.title("Project Type Distribution")
plt.xlabel("Project Type")
plt.ylabel("Number of Projects")
plt.show()


# In[12]:


df["Methodology_Used"].value_counts().plot(kind="bar")
plt.title("Methodology Distribution")
plt.xlabel("Methodology")
plt.ylabel("Number of Projects")
plt.show()


# In[13]:


agile_it_df = df[
    (df["Project_Type"] == "IT") &
    (df["Methodology_Used"].isin(["Agile", "Scrum", "Kanban"]))
].copy()

print("Agile/IT subset shape:", agile_it_df.shape)
print("\nRisk level distribution in Agile/IT subset:")
print(agile_it_df["Risk_Level"].value_counts())

print("\nBinary target distribution in Agile/IT subset:")
print(agile_it_df["Elevated_Risk"].value_counts())


# In[14]:


comparison = pd.DataFrame({
    "Dataset": ["Full Dataset", "Agile/IT Subset"],
    "Rows": [df.shape[0], agile_it_df.shape[0]],
    "Columns": [df.shape[1], agile_it_df.shape[1]],
    "Elevated Risk Count": [
        df["Elevated_Risk"].sum(),
        agile_it_df["Elevated_Risk"].sum()
    ],
    "Elevated Risk %": [
        round(df["Elevated_Risk"].mean() * 100, 2),
        round(agile_it_df["Elevated_Risk"].mean() * 100, 2)
    ]
})

comparison


# In[15]:


numeric_columns = agile_it_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

numeric_columns


# In[16]:


agile_it_df[numeric_columns].describe().T


# In[17]:


risk_by_methodology = pd.crosstab(
    agile_it_df["Methodology_Used"],
    agile_it_df["Risk_Level"],
    normalize="index"
) * 100

risk_by_methodology


# In[18]:


risk_by_methodology.plot(kind="bar")
plt.title("Risk Level Distribution by Agile Methodology")
plt.xlabel("Methodology")
plt.ylabel("Percentage")
plt.legend(title="Risk Level")
plt.show()


# In[19]:


agile_it_df.to_csv("agile_it_project_risk_dataset.csv", index=False)

print("Clean Agile/IT dataset saved successfully.")


# In[20]:


df.shape
df["Risk_Level"].value_counts()
agile_it_df.shape
agile_it_df["Elevated_Risk"].value_counts()


# In[21]:


# Define the target variable
y = agile_it_df["Elevated_Risk"]

# Drop columns that should not be used as predictors
X = agile_it_df.drop(columns=[
    "Risk_Level",       # original target column
    "Elevated_Risk",    # engineered target column
    "Project_ID"        # identifier, not useful for prediction
])

print("Feature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)


# In[22]:


numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", len(numeric_features))
print(numeric_features)

print("\nCategorical features:", len(categorical_features))
print(categorical_features)


# In[23]:


from pandas.api.types import is_numeric_dtype

numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
categorical_features = [col for col in X.columns if col not in numeric_features]

print("Numeric features:", len(numeric_features))
print(numeric_features)

print("\nCategorical features:", len(categorical_features))
print(categorical_features)


# In[24]:


print("Total selected columns:", len(numeric_features) + len(categorical_features))
print("Total X columns:", X.shape[1])


# In[25]:


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully.")


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))


# In[27]:


log_reg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

log_reg_model.fit(X_train, y_train)

y_pred_lr = log_reg_model.predict(X_test)
y_prob_lr = log_reg_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("Precision:", round(precision_score(y_test, y_pred_lr), 3))
print("Recall:", round(recall_score(y_test, y_pred_lr), 3))
print("F1-score:", round(f1_score(y_test, y_pred_lr), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_lr), 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))


# In[28]:


rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 3))
print("Precision:", round(precision_score(y_test, y_pred_rf), 3))
print("Recall:", round(recall_score(y_test, y_pred_rf), 3))
print("F1-score:", round(f1_score(y_test, y_pred_rf), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_rf), 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# In[29]:


from xgboost import XGBClassifier

xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred_xgb), 3))
print("Precision:", round(precision_score(y_test, y_pred_xgb), 3))
print("Recall:", round(recall_score(y_test, y_pred_xgb), 3))
print("F1-score:", round(f1_score(y_test, y_pred_xgb), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob_xgb), 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))


# In[30]:


model_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    "F1-score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_xgb)
    ]
})

model_results = model_results.round(3)
model_results


# In[31]:


model_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].plot(kind="bar")
plt.title("Preliminary Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.show()


# In[32]:


model_results.to_csv("preliminary_model_results.csv", index=False)

print("Preliminary model results saved successfully.")


# In[33]:


from sklearn.metrics import ConfusionMatrixDisplay

models_predictions = {
    "Logistic Regression": y_pred_lr,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}

for model_name, predictions in models_predictions.items():
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Elevated Risk", "Elevated Risk"])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


# In[34]:


from sklearn.model_selection import StratifiedKFold, cross_validate

models = {
    "Logistic Regression": log_reg_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results_list = []

for model_name, model in models.items():
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    cv_results_list.append({
        "Model": model_name,
        "Accuracy Mean": results["test_accuracy"].mean(),
        "Precision Mean": results["test_precision"].mean(),
        "Recall Mean": results["test_recall"].mean(),
        "F1 Mean": results["test_f1"].mean(),
        "ROC-AUC Mean": results["test_roc_auc"].mean()
    })

cv_results = pd.DataFrame(cv_results_list).round(3)
cv_results


# In[35]:


cv_results.to_csv("cross_validation_results.csv", index=False)


# In[36]:


# Get transformed feature names from preprocessing pipeline
feature_names = log_reg_model.named_steps["preprocessor"].get_feature_names_out()

# Get Logistic Regression coefficients
coefficients = log_reg_model.named_steps["classifier"].coef_[0]

logistic_feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients)
}).sort_values(by="Absolute_Coefficient", ascending=False)

logistic_feature_importance.head(20)


# In[37]:


top_logistic_features = logistic_feature_importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_logistic_features["Feature"], top_logistic_features["Coefficient"])
plt.title("Top Logistic Regression Predictors of Elevated Risk")
plt.xlabel("Coefficient Value")
plt.gca().invert_yaxis()
plt.show()


# In[38]:


rf_feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.named_steps["classifier"].feature_importances_
}).sort_values(by="Importance", ascending=False)

rf_feature_importance.head(20)


# In[39]:


top_rf_features = rf_feature_importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_rf_features["Feature"], top_rf_features["Importance"])
plt.title("Top Random Forest Feature Importances")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()


# In[40]:


xgb_feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_model.named_steps["classifier"].feature_importances_
}).sort_values(by="Importance", ascending=False)

xgb_feature_importance.head(20)


# In[41]:


top_xgb_features = xgb_feature_importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_xgb_features["Feature"], top_xgb_features["Importance"])
plt.title("Top XGBoost Feature Importances")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()


# In[42]:


X.columns.tolist()


# In[43]:


clean_logistic_features = logistic_feature_importance.copy()

clean_logistic_features["Feature_Clean"] = (
    clean_logistic_features["Feature"]
    .str.replace("cat__", "", regex=False)
    .str.replace("num__", "", regex=False)
    .str.replace("_", " ", regex=False)
)

clean_logistic_features["Direction"] = clean_logistic_features["Coefficient"].apply(
    lambda x: "Increases elevated risk" if x > 0 else "Reduces elevated risk"
)

clean_logistic_features[[
    "Feature_Clean",
    "Coefficient",
    "Direction"
]].head(20)


# In[44]:


clean_logistic_features[[
    "Feature_Clean",
    "Coefficient",
    "Absolute_Coefficient",
    "Direction"
]].head(20).to_csv("top_20_logistic_regression_predictors.csv", index=False)

print("Top predictor table saved.")


# In[45]:


top_15_clean = clean_logistic_features.head(15).sort_values("Coefficient")

plt.figure(figsize=(10, 7))
plt.barh(top_15_clean["Feature_Clean"], top_15_clean["Coefficient"])
plt.axvline(0)
plt.title("Top Logistic Regression Predictors of Elevated Project Risk")
plt.xlabel("Coefficient Value")
plt.ylabel("Project Risk Feature")
plt.tight_layout()
plt.show()


# In[46]:


from sklearn.model_selection import StratifiedKFold, cross_validate

models = {
    "Logistic Regression": log_reg_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results_list = []

for model_name, model in models.items():
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    cv_results_list.append({
        "Model": model_name,
        "Accuracy Mean": results["test_accuracy"].mean(),
        "Precision Mean": results["test_precision"].mean(),
        "Recall Mean": results["test_recall"].mean(),
        "F1 Mean": results["test_f1"].mean(),
        "ROC-AUC Mean": results["test_roc_auc"].mean()
    })

cv_results = pd.DataFrame(cv_results_list).round(3)
cv_results


# In[1]:


print(X.shape)
print(y.shape)
print(X_train.shape)
print(X_test.shape)

print(model_results)
print(cv_results)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("project_risk_raw_dataset.csv")


# In[3]:


df["Elevated_Risk"] = df["Risk_Level"].isin(["High", "Critical"]).astype(int)


# In[4]:


agile_it_df = df[
    (df["Project_Type"] == "IT") &
    (df["Methodology_Used"].isin(["Agile", "Scrum", "Kanban"]))
].copy()


# In[5]:


y = agile_it_df["Elevated_Risk"]

X = agile_it_df.drop(columns=[
    "Risk_Level",
    "Elevated_Risk",
    "Project_ID"
])

print("X shape:", X.shape)
print("y shape:", y.shape)


# In[6]:


from pandas.api.types import is_numeric_dtype

numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
categorical_features = [col for col in X.columns if col not in numeric_features]


# In[7]:


print(X.shape)
print(y.shape)
print(X_train.shape)
print(X_test.shape)

print(model_results)
print(cv_results)


# In[8]:


from pandas.api.types import is_numeric_dtype

numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
categorical_features = [col for col in X.columns if col not in numeric_features]


# In[9]:


X.columns.tolist()


# In[10]:


for col in X.columns:
    print(col)


# In[11]:


suspected_leakage = [
    col for col in X.columns
    if any(word in col.lower() for word in [
        "risk", "delay", "outcome", "success", "status"
    ])
]

suspected_leakage


# In[12]:


from sklearn.dummy import DummyClassifier


# In[13]:


dummy_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

dummy_model.fit(X_train, y_train)

y_pred_dummy = dummy_model.predict(X_test)
y_prob_dummy = dummy_model.predict_proba(X_test)[:, 1]

dummy_results = {
    "Accuracy": accuracy_score(y_test, y_pred_dummy),
    "Precision": precision_score(y_test, y_pred_dummy, zero_division=0),
    "Recall": recall_score(y_test, y_pred_dummy, zero_division=0),
    "F1": f1_score(y_test, y_pred_dummy, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, y_prob_dummy)
}

dummy_results


# In[14]:


# Core Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Identify numerical and categorical features
from pandas.api.types import is_numeric_dtype

numeric_features = [
    col for col in X.columns 
    if is_numeric_dtype(X[col])
]

categorical_features = [
    col for col in X.columns 
    if col not in numeric_features
]

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Pipeline setup completed.")
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# In[16]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

dummy_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

dummy_model.fit(X_train, y_train)

y_pred_dummy = dummy_model.predict(X_test)
y_prob_dummy = dummy_model.predict_proba(X_test)[:, 1]

dummy_results = {
    "Accuracy": accuracy_score(y_test, y_pred_dummy),
    "Precision": precision_score(y_test, y_pred_dummy, zero_division=0),
    "Recall": recall_score(y_test, y_pred_dummy, zero_division=0),
    "F1": f1_score(y_test, y_pred_dummy, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob_dummy)
}

dummy_results


# In[17]:


from sklearn.model_selection import GridSearchCV


# In[18]:


log_reg_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

log_reg_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

log_reg_grid = GridSearchCV(
    estimator=log_reg_tuning_pipeline,
    param_grid=log_reg_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

log_reg_grid.fit(X_train, y_train)

print("Best parameters:", log_reg_grid.best_params_)
print("Best CV ROC-AUC:", log_reg_grid.best_score_)


# In[22]:


log_reg_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

log_reg_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

log_reg_grid = GridSearchCV(
    estimator=log_reg_tuning_pipeline,
    param_grid=log_reg_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

log_reg_grid.fit(X_train, y_train)

print("Best parameters:", log_reg_grid.best_params_)
print("Best CV ROC-AUC:", log_reg_grid.best_score_)


# In[23]:


best_log_reg = log_reg_grid.best_estimator_


# In[24]:


rf_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

rf_params = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

rf_grid = GridSearchCV(
    estimator=rf_tuning_pipeline,
    param_grid=rf_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best parameters:", rf_grid.best_params_)
print("Best CV ROC-AUC:", rf_grid.best_score_)


# In[25]:


best_rf = rf_grid.best_estimator_


# In[26]:


from sklearn.model_selection import RandomizedSearchCV


# In[27]:


xgb_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_params = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [2, 3, 4, 5, 6],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.8, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_tuning_pipeline,
    param_distributions=xgb_params,
    n_iter=25,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X_train, y_train)

print("Best parameters:", xgb_random.best_params_)
print("Best CV ROC-AUC:", xgb_random.best_score_)


# In[28]:


best_xgb = xgb_random.best_estimator_


# In[29]:


tuned_models = {
    "Logistic Regression": best_log_reg,
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

final_results = []

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    final_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

final_results_df = pd.DataFrame(final_results).round(3)
final_results_df


# In[30]:


from sklearn.model_selection import RepeatedStratifiedKFold


# In[31]:


repeated_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# In[32]:


robust_cv_results = []

for name, model in tuned_models.items():
    results = cross_validate(
        model,
        X,
        y,
        cv=repeated_cv,
        scoring=scoring,
        n_jobs=-1
    )

    robust_cv_results.append({
        "Model": name,
        "Accuracy Mean": results["test_accuracy"].mean(),
        "Accuracy SD": results["test_accuracy"].std(),
        "Precision Mean": results["test_precision"].mean(),
        "Recall Mean": results["test_recall"].mean(),
        "F1 Mean": results["test_f1"].mean(),
        "ROC-AUC Mean": results["test_roc_auc"].mean(),
        "ROC-AUC SD": results["test_roc_auc"].std()
    })

robust_cv_results_df = pd.DataFrame(robust_cv_results).round(3)
robust_cv_results_df


# In[33]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[34]:


for name, model in tuned_models.items():
    y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Non-Elevated", "Elevated"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.show()


# In[35]:


from sklearn.metrics import RocCurveDisplay


# In[36]:


plt.figure(figsize=(8, 6))

for name, model in tuned_models.items():
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name
    )

plt.title("ROC Curves - Tuned Models")
plt.show()


# In[37]:


from sklearn.metrics import PrecisionRecallDisplay


# In[38]:


plt.figure(figsize=(8, 6))

for name, model in tuned_models.items():
    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name
    )

plt.title("Precision-Recall Curves - Tuned Models")
plt.show()


# In[39]:


cost_results = []

false_positive_cost = 1
false_negative_cost = 5

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_cost = (
        fp * false_positive_cost +
        fn * false_negative_cost
    )

    cost_results.append({
        "Model": name,
        "False Positives": fp,
        "False Negatives": fn,
        "Weighted Error Cost": total_cost
    })

cost_results_df = pd.DataFrame(cost_results)
cost_results_df


# In[40]:


feature_names = best_log_reg.named_steps["preprocessor"].get_feature_names_out()

coefficients = best_log_reg.named_steps["classifier"].coef_[0]

lr_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients)
}).sort_values("Absolute_Coefficient", ascending=False)

lr_importance.head(20)


# In[41]:


rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": best_rf.named_steps["classifier"].feature_importances_
}).sort_values("Importance", ascending=False)

rf_importance.head(20)


# In[42]:


xgb_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": best_xgb.named_steps["classifier"].feature_importances_
}).sort_values("Importance", ascending=False)

xgb_importance.head(20)


# In[43]:


print("Dataset:", df.shape)
print("Agile/IT subset:", agile_it_df.shape)
print("X:", X.shape)
print("y:", y.shape)
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

print("\nTarget distribution:")
print(y.value_counts())


# In[44]:


suspected_leakage = [
    col for col in X.columns
    if any(
        word in col.lower()
        for word in [
            "risk",
            "delay",
            "outcome",
            "success",
            "status"
        ]
    )
]

print("Potential leakage-related columns:")
for col in suspected_leakage:
    print("-", col)


# In[45]:


print(X.columns.tolist())


# In[46]:


print(X.columns.tolist())


# In[47]:


dummy_model.fit(X_train, y_train)

y_pred_dummy = dummy_model.predict(X_test)
y_prob_dummy = dummy_model.predict_proba(X_test)[:, 1]

dummy_results = {
    "Accuracy": round(
        accuracy_score(y_test, y_pred_dummy), 3
    ),
    "Precision": round(
        precision_score(y_test, y_pred_dummy, zero_division=0), 3
    ),
    "Recall": round(
        recall_score(y_test, y_pred_dummy, zero_division=0), 3
    ),
    "F1": round(
        f1_score(y_test, y_pred_dummy, zero_division=0), 3
    ),
    "ROC-AUC": round(
        roc_auc_score(y_test, y_prob_dummy), 3
    )
}

dummy_results


# In[48]:


baseline_models = {
    "Logistic Regression": log_reg_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

baseline_results = []

for name, model in baseline_models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    baseline_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

baseline_results_df = pd.DataFrame(baseline_results).round(3)

baseline_results_df


# In[49]:


log_reg_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

log_reg_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

log_reg_grid = GridSearchCV(
    estimator=log_reg_tuning_pipeline,
    param_grid=log_reg_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

log_reg_grid.fit(X_train, y_train)

print("Best parameters:")
print(log_reg_grid.best_params_)

print("\nBest CV ROC-AUC:")
print(round(log_reg_grid.best_score_, 4))

best_log_reg = log_reg_grid.best_estimator_


# In[50]:


rf_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

rf_params = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 5, 10, 15, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"]
}

rf_random = RandomizedSearchCV(
    estimator=rf_tuning_pipeline,
    param_distributions=rf_params,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)

print("Best parameters:")
print(rf_random.best_params_)

print("\nBest CV ROC-AUC:")
print(round(rf_random.best_score_, 4))

best_rf = rf_random.best_estimator_


# In[51]:


xgb_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_params = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [2, 3, 4, 5, 6],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.8, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_tuning_pipeline,
    param_distributions=xgb_params,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X_train, y_train)

print("Best parameters:")
print(xgb_random.best_params_)

print("\nBest CV ROC-AUC:")
print(round(xgb_random.best_score_, 4))

best_xgb = xgb_random.best_estimator_


# In[52]:


tuned_models = {
    "Logistic Regression": best_log_reg,
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

final_results = []

for name, model in tuned_models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    final_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

final_results_df = pd.DataFrame(final_results).round(3)

final_results_df


# In[53]:


final_results_df.to_csv(
    "final_tuned_model_results.csv",
    index=False
)


# In[54]:


repeated_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# In[55]:


robust_cv_results = []

for name, model in tuned_models.items():

    results = cross_validate(
        model,
        X,
        y,
        cv=repeated_cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    robust_cv_results.append({
        "Model": name,

        "Accuracy Mean":
            results["test_accuracy"].mean(),

        "Accuracy SD":
            results["test_accuracy"].std(),

        "Precision Mean":
            results["test_precision"].mean(),

        "Recall Mean":
            results["test_recall"].mean(),

        "F1 Mean":
            results["test_f1"].mean(),

        "ROC-AUC Mean":
            results["test_roc_auc"].mean(),

        "ROC-AUC SD":
            results["test_roc_auc"].std()
    })

robust_cv_results_df = (
    pd.DataFrame(robust_cv_results)
    .round(3)
)

robust_cv_results_df


# In[56]:


robust_cv_results_df.to_csv(
    "repeated_cross_validation_results.csv",
    index=False
)


# In[57]:


for name, model in tuned_models.items():

    y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Non-Elevated",
            "Elevated"
        ]
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.show()


# In[58]:


fig, ax = plt.subplots(figsize=(8, 6))

for name, model in tuned_models.items():

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax.set_title(
    "ROC Curves - Tuned Models"
)

plt.show()


# In[59]:


fig, ax = plt.subplots(figsize=(8, 6))

for name, model in tuned_models.items():

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

ax.set_title(
    "Precision-Recall Curves - Tuned Models"
)

plt.show()


# In[60]:


cost_scenarios = {
    "Equal Cost": 1,
    "Moderate FN Cost": 3,
    "High FN Cost": 5,
    "Severe FN Cost": 10
}

cost_results = []

for name, model in tuned_models.items():

    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    for scenario, fn_cost in cost_scenarios.items():

        fp_cost = 1

        total_cost = (
            fp * fp_cost +
            fn * fn_cost
        )

        cost_results.append({
            "Model": name,
            "Scenario": scenario,
            "False Positives": fp,
            "False Negatives": fn,
            "Weighted Cost":
                total_cost
        })

cost_results_df = pd.DataFrame(
    cost_results
)

cost_results_df


# In[61]:


best_model = best_log_reg

y_prob = best_model.predict_proba(
    X_test
)[:, 1]


# In[62]:


thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

threshold_results = []

for threshold in thresholds:

    predictions = (
        y_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions
            ),

        "F1":
            f1_score(
                y_test,
                predictions
            ),

        "False Positives": fp,
        "False Negatives": fn
    })

threshold_results_df = (
    pd.DataFrame(threshold_results)
    .round(3)
)

threshold_results_df


# In[63]:


feature_names = (
    best_log_reg
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    best_log_reg
    .named_steps["classifier"]
    .coef_[0]
)

lr_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient":
        np.abs(coefficients)
}).sort_values(
    "Absolute_Coefficient",
    ascending=False
)

lr_importance.head(20)


# In[64]:


lr_importance["Odds_Ratio"] = np.exp(
    lr_importance["Coefficient"]
)

lr_importance.head(20)


# In[65]:


rf_feature_names = (
    best_rf
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

rf_importance = pd.DataFrame({
    "Feature": rf_feature_names,

    "Importance":
        best_rf
        .named_steps["classifier"]
        .feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

rf_importance.head(20)


# In[66]:


xgb_feature_names = (
    best_xgb
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

xgb_importance = pd.DataFrame({
    "Feature": xgb_feature_names,

    "Importance":
        best_xgb
        .named_steps["classifier"]
        .feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

xgb_importance.head(20)


# In[67]:


import shap


# In[68]:


import sys
print(sys.executable)


# In[69]:


import shap
print(shap.__version__)


# In[70]:


conda activate agile-risk
pip install --upgrade shap


# In[71]:


import shap


# In[72]:


preprocessor_xgb = (
    best_xgb.named_steps[
        "preprocessor"
    ]
)

xgb_classifier = (
    best_xgb.named_steps[
        "classifier"
    ]
)

X_test_transformed = (
    preprocessor_xgb
    .transform(X_test)
)

if hasattr(
    X_test_transformed,
    "toarray"
):
    X_test_transformed = (
        X_test_transformed
        .toarray()
    )

feature_names_xgb = (
    preprocessor_xgb
    .get_feature_names_out()
)


# In[73]:


explainer = shap.TreeExplainer(
    xgb_classifier
)

shap_values = explainer.shap_values(
    X_test_transformed
)

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=
        feature_names_xgb
)


# In[74]:


    y_multi = agile_it_df[
    "Risk_Level"
]


# In[75]:


y_multi = agile_it_df[
    "Risk_Level"
]


# In[76]:


X_multi = agile_it_df.drop(
    columns=[
        "Risk_Level",
        "Elevated_Risk",
        "Project_ID"
    ]
)


# In[77]:


Main Agile/IT feature set

vs

Conservative feature set
(with questionable predictors removed)


# In[78]:


Main Agile/IT feature set

vs

Conservative feature set
(with questionable predictors removed)


# In[79]:


candidate_columns = [
    "Historical_Risk_Incidents",
    "Previous_Delivery_Success_Rate",
    "Risk_Management_Maturity",
    "Seasonal_Risk_Factor"
]

for col in candidate_columns:
    print("=" * 70)
    print("COLUMN:", col)
    print("Data type:", agile_it_df[col].dtype)
    print("Missing values:", agile_it_df[col].isnull().sum())
    print("Unique values:", agile_it_df[col].nunique())

    if pd.api.types.is_numeric_dtype(agile_it_df[col]):
        print("\nSummary:")
        print(agile_it_df[col].describe())

        print("\nMean by Elevated_Risk:")
        print(
            agile_it_df.groupby("Elevated_Risk")[col].mean()
        )
    else:
        print("\nValues:")
        print(agile_it_df[col].value_counts(dropna=False))


# In[80]:


log_reg_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

log_reg_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

log_reg_grid = GridSearchCV(
    estimator=log_reg_tuning_pipeline,
    param_grid=log_reg_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

log_reg_grid.fit(X_train, y_train)

print("Best Logistic Regression parameters:")
print(log_reg_grid.best_params_)

print("\nBest Logistic Regression CV ROC-AUC:")
print(round(log_reg_grid.best_score_, 4))

best_log_reg = log_reg_grid.best_estimator_


# In[81]:


y_pred_lr_tuned = best_log_reg.predict(X_test)
y_prob_lr_tuned = best_log_reg.predict_proba(X_test)[:, 1]

tuned_lr_results = {
    "Accuracy": round(accuracy_score(y_test, y_pred_lr_tuned), 3),
    "Precision": round(precision_score(y_test, y_pred_lr_tuned), 3),
    "Recall": round(recall_score(y_test, y_pred_lr_tuned), 3),
    "F1": round(f1_score(y_test, y_pred_lr_tuned), 3),
    "ROC-AUC": round(roc_auc_score(y_test, y_prob_lr_tuned), 3)
}

tuned_lr_results


# In[82]:


rf_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

rf_params = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 5, 10, 15, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"]
}

rf_random = RandomizedSearchCV(
    estimator=rf_tuning_pipeline,
    param_distributions=rf_params,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)

print("Best Random Forest parameters:")
print(rf_random.best_params_)

print("\nBest Random Forest CV ROC-AUC:")
print(round(rf_random.best_score_, 4))

best_rf = rf_random.best_estimator_


# In[83]:


xgb_tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

xgb_params = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [2, 3, 4, 5, 6],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.8, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_tuning_pipeline,
    param_distributions=xgb_params,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X_train, y_train)

print("Best XGBoost parameters:")
print(xgb_random.best_params_)

print("\nBest XGBoost CV ROC-AUC:")
print(round(xgb_random.best_score_, 4))

best_xgb = xgb_random.best_estimator_


# In[84]:


tuned_models = {
    "Logistic Regression": best_log_reg,
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

final_results = []

for name, model in tuned_models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    final_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    })

final_results_df = pd.DataFrame(
    final_results
).round(3)

final_results_df


# In[85]:


final_results_df.to_csv(
    "final_tuned_model_results.csv",
    index=False
)


# In[86]:


tuning_comparison = baseline_results_df.merge(
    final_results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    ],
    on="Model",
    suffixes=("_Baseline", "_Tuned")
)

tuning_comparison


# In[87]:


tuning_comparison.to_csv(
    "baseline_vs_tuned_results.csv",
    index=False
)


# In[88]:


from sklearn.model_selection import RepeatedStratifiedKFold

repeated_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# In[89]:


robust_cv_results = []
cv_score_storage = {}

for name, model in tuned_models.items():

    results = cross_validate(
        model,
        X,
        y,
        cv=repeated_cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    # Save individual fold scores for later
    # statistical comparison
    cv_score_storage[name] = results

    robust_cv_results.append({
        "Model": name,

        "Accuracy Mean":
            results["test_accuracy"].mean(),

        "Accuracy SD":
            results["test_accuracy"].std(),

        "Precision Mean":
            results["test_precision"].mean(),

        "Precision SD":
            results["test_precision"].std(),

        "Recall Mean":
            results["test_recall"].mean(),

        "Recall SD":
            results["test_recall"].std(),

        "F1 Mean":
            results["test_f1"].mean(),

        "F1 SD":
            results["test_f1"].std(),

        "ROC-AUC Mean":
            results["test_roc_auc"].mean(),

        "ROC-AUC SD":
            results["test_roc_auc"].std()
    })

robust_cv_results_df = (
    pd.DataFrame(robust_cv_results)
    .round(3)
)

robust_cv_results_df


# In[90]:


robust_cv_results_df.to_csv(
    "repeated_cross_validation_results.csv",
    index=False
)


# In[91]:


from scipy.stats import wilcoxon


# In[92]:


model_pairs = [
    ("Logistic Regression", "Random Forest"),
    ("Logistic Regression", "XGBoost"),
    ("Random Forest", "XGBoost")
]

statistical_results = []

for model_a, model_b in model_pairs:

    scores_a = (
        cv_score_storage[model_a]
        ["test_roc_auc"]
    )

    scores_b = (
        cv_score_storage[model_b]
        ["test_roc_auc"]
    )

    statistic, p_value = wilcoxon(
        scores_a,
        scores_b
    )

    statistical_results.append({
        "Model A": model_a,
        "Model B": model_b,
        "Wilcoxon Statistic": statistic,
        "p-value": p_value
    })

statistical_results_df = (
    pd.DataFrame(statistical_results)
    .round(5)
)

statistical_results_df


# In[93]:


statistical_results_df.to_csv(
    "statistical_model_comparison.csv",
    index=False
)


# In[94]:


for name, model in tuned_models.items():

    y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Non-Elevated",
            "Elevated"
        ]
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.tight_layout()
    plt.show()


# In[95]:


fig, ax = plt.subplots(figsize=(8, 6))

for name, model in tuned_models.items():

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax.set_title(
    "ROC Curves - Final Tuned Models"
)

plt.tight_layout()
plt.show()


# In[96]:


fig, ax = plt.subplots(figsize=(8, 6))

for name, model in tuned_models.items():

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )

ax.set_title(
    "Precision-Recall Curves - Final Tuned Models"
)

plt.tight_layout()
plt.show()


# In[97]:


cost_scenarios = {
    "Equal (FN = 1)": 1,
    "Moderate (FN = 3)": 3,
    "High (FN = 5)": 5,
    "Severe (FN = 10)": 10
}

cost_results = []

for name, model in tuned_models.items():

    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    for scenario, fn_cost in cost_scenarios.items():

        fp_cost = 1

        weighted_cost = (
            fp * fp_cost +
            fn * fn_cost
        )

        cost_results.append({
            "Model": name,
            "Scenario": scenario,
            "FP": fp,
            "FN": fn,
            "Weighted Error Cost":
                weighted_cost
        })

cost_results_df = pd.DataFrame(
    cost_results
)

cost_results_df


# In[98]:


cost_results_df.to_csv(
    "cost_sensitive_analysis.csv",
    index=False
)


# In[99]:


best_model = best_log_reg

y_prob_best = best_model.predict_proba(
    X_test
)[:, 1]

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

threshold_results = []

for threshold in thresholds:

    predictions = (
        y_prob_best >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),
        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
        "Recall":
            recall_score(
                y_test,
                predictions
            ),
        "F1":
            f1_score(
                y_test,
                predictions
            ),
        "FP": fp,
        "FN": fn
    })

threshold_results_df = (
    pd.DataFrame(threshold_results)
    .round(3)
)

threshold_results_df


# In[100]:


from sklearn.model_selection import cross_val_predict


# In[101]:


threshold_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_probabilities = cross_val_predict(
    best_log_reg,
    X_train,
    y_train,
    cv=threshold_cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


# In[102]:


thresholds = np.arange(
    0.20,
    0.81,
    0.05
)

threshold_validation_results = []

for threshold in thresholds:

    predictions = (
        train_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train,
        predictions
    ).ravel()

    threshold_validation_results.append({
        "Threshold": round(threshold, 2),
        "Accuracy":
            accuracy_score(
                y_train,
                predictions
            ),
        "Precision":
            precision_score(
                y_train,
                predictions,
                zero_division=0
            ),
        "Recall":
            recall_score(
                y_train,
                predictions
            ),
        "F1":
            f1_score(
                y_train,
                predictions
            ),
        "FP": fp,
        "FN": fn
    })

threshold_validation_df = (
    pd.DataFrame(
        threshold_validation_results
    )
    .round(3)
)

threshold_validation_df


# In[103]:


best_threshold_row = (
    threshold_validation_df
    .loc[
        threshold_validation_df[
            "F1"
        ].idxmax()
    ]
)

best_threshold_row


# In[104]:


selected_threshold = float(
    best_threshold_row["Threshold"]
)

print(
    "Selected threshold:",
    selected_threshold
)


# In[105]:


final_test_probabilities = (
    best_log_reg
    .predict_proba(X_test)[:, 1]
)

final_threshold_predictions = (
    final_test_probabilities
    >= selected_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    final_threshold_predictions
).ravel()

final_threshold_result = {
    "Selected Threshold":
        selected_threshold,

    "Accuracy":
        round(
            accuracy_score(
                y_test,
                final_threshold_predictions
            ),
            3
        ),

    "Precision":
        round(
            precision_score(
                y_test,
                final_threshold_predictions
            ),
            3
        ),

    "Recall":
        round(
            recall_score(
                y_test,
                final_threshold_predictions
            ),
            3
        ),

    "F1":
        round(
            f1_score(
                y_test,
                final_threshold_predictions
            ),
            3
        ),

    "False Positives": fp,
    "False Negatives": fn,
    "True Positives": tp,
    "True Negatives": tn
}

final_threshold_result


# In[106]:


feature_names = (
    best_log_reg
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    best_log_reg
    .named_steps["classifier"]
    .coef_[0]
)

lr_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients),
    "Odds_Ratio": np.exp(coefficients)
}).sort_values(
    "Absolute_Coefficient",
    ascending=False
)

lr_importance.head(20)


# In[107]:


lr_importance["Feature_Clean"] = (
    lr_importance["Feature"]
    .str.replace("cat__", "", regex=False)
    .str.replace("num__", "", regex=False)
    .str.replace("_", " ", regex=False)
)

lr_importance["Direction"] = np.where(
    lr_importance["Coefficient"] > 0,
    "Higher elevated-risk likelihood",
    "Lower elevated-risk likelihood"
)

lr_importance[
    [
        "Feature_Clean",
        "Coefficient",
        "Odds_Ratio",
        "Direction"
    ]
].head(20)


# In[108]:


rf_feature_names = (
    best_rf
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

rf_importance = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance":
        best_rf
        .named_steps["classifier"]
        .feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

rf_importance.head(20)


# In[109]:


xgb_feature_names = (
    best_xgb
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

xgb_importance = pd.DataFrame({
    "Feature": xgb_feature_names,
    "Importance":
        best_xgb
        .named_steps["classifier"]
        .feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

xgb_importance.head(20)


# In[110]:


def get_original_feature(transformed_name):

    # Numeric features
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "")

    # Categorical features
    if transformed_name.startswith("cat__"):

        clean_name = transformed_name.replace("cat__", "")

        # Match against the known original
        # categorical column names
        for col in sorted(
            categorical_features,
            key=len,
            reverse=True
        ):
            prefix = col + "_"

            if clean_name.startswith(prefix):
                return col

        return clean_name

    return transformed_name


# In[111]:


lr_importance["Original_Feature"] = (
    lr_importance["Feature"]
    .apply(get_original_feature)
)

rf_importance["Original_Feature"] = (
    rf_importance["Feature"]
    .apply(get_original_feature)
)

xgb_importance["Original_Feature"] = (
    xgb_importance["Feature"]
    .apply(get_original_feature)
)


# In[112]:


lr_grouped = (
    lr_importance
    .groupby("Original_Feature")
    .agg(
        LR_Importance=(
            "Coefficient",
            lambda x: np.sqrt(
                np.sum(np.square(x))
            )
        )
    )
    .reset_index()
    .sort_values(
        "LR_Importance",
        ascending=False
    )
)

lr_grouped.head(15)


# In[113]:


rf_grouped = (
    rf_importance
    .groupby("Original_Feature")
    ["Importance"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "Importance":
            "RF_Importance"
        }
    )
    .sort_values(
        "RF_Importance",
        ascending=False
    )
)

rf_grouped.head(15)


# In[114]:


xgb_grouped = (
    xgb_importance
    .groupby("Original_Feature")
    ["Importance"]
    .sum()
    .reset_index()
    .rename(
        columns={
            "Importance":
            "XGB_Importance"
        }
    )
    .sort_values(
        "XGB_Importance",
        ascending=False
    )
)

xgb_grouped.head(15)


# In[115]:


feature_comparison = (
    lr_grouped
    .merge(
        rf_grouped,
        on="Original_Feature",
        how="outer"
    )
    .merge(
        xgb_grouped,
        on="Original_Feature",
        how="outer"
    )
    .fillna(0)
)


# In[116]:


for col in [
    "LR_Importance",
    "RF_Importance",
    "XGB_Importance"
]:
    feature_comparison[
        col + "_Norm"
    ] = (
        feature_comparison[col] /
        feature_comparison[col].max()
    )


# In[117]:


feature_comparison[
    "Mean_Normalized_Importance"
] = feature_comparison[
    [
        "LR_Importance_Norm",
        "RF_Importance_Norm",
        "XGB_Importance_Norm"
    ]
].mean(axis=1)

cross_model_features = (
    feature_comparison
    .sort_values(
        "Mean_Normalized_Importance",
        ascending=False
    )
)

cross_model_features[
    [
        "Original_Feature",
        "LR_Importance_Norm",
        "RF_Importance_Norm",
        "XGB_Importance_Norm",
        "Mean_Normalized_Importance"
    ]
].head(15)


# In[118]:


cross_model_features[
    "LR_Rank"
] = cross_model_features[
    "LR_Importance"
].rank(
    ascending=False,
    method="min"
)

cross_model_features[
    "RF_Rank"
] = cross_model_features[
    "RF_Importance"
].rank(
    ascending=False,
    method="min"
)

cross_model_features[
    "XGB_Rank"
] = cross_model_features[
    "XGB_Importance"
].rank(
    ascending=False,
    method="min"
)

cross_model_features[
    "Mean_Rank"
] = cross_model_features[
    [
        "LR_Rank",
        "RF_Rank",
        "XGB_Rank"
    ]
].mean(axis=1)

cross_model_features.sort_values(
    "Mean_Rank"
)[
    [
        "Original_Feature",
        "LR_Rank",
        "RF_Rank",
        "XGB_Rank",
        "Mean_Rank"
    ]
].head(15)


# In[119]:


cross_model_features.to_csv(
    "cross_model_feature_comparison.csv",
    index=False
)


# In[120]:


import shap
print("SHAP version:", shap.__version__)


# In[121]:


preprocessor_xgb = best_xgb.named_steps["preprocessor"]
xgb_classifier = best_xgb.named_steps["classifier"]

X_test_transformed = preprocessor_xgb.transform(X_test)

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

feature_names_xgb = preprocessor_xgb.get_feature_names_out()

print("Transformed test shape:", X_test_transformed.shape)
print("Feature names:", len(feature_names_xgb))


# In[122]:


explainer = shap.TreeExplainer(xgb_classifier)

shap_values = explainer.shap_values(X_test_transformed)

print("SHAP values calculated.")


# In[123]:


shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names_xgb
)


# In[124]:


shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names_xgb,
    plot_type="bar"
)


# In[125]:


xgb_test_predictions = best_xgb.predict(X_test)
xgb_test_probabilities = best_xgb.predict_proba(X_test)[:, 1]

correct_high_risk_positions = np.where(
    (y_test.to_numpy() == 1) &
    (xgb_test_predictions == 1)
)[0]

example_position = correct_high_risk_positions[0]

print("Example position:", example_position)
print(
    "Predicted elevated-risk probability:",
    round(xgb_test_probabilities[example_position], 3)
)
print(
    "Actual class:",
    y_test.iloc[example_position]
)


# In[126]:


shap_explanation = shap.Explanation(
    values=shap_values[example_position],
    base_values=explainer.expected_value,
    data=X_test_transformed[example_position],
    feature_names=feature_names_xgb
)

shap.plots.waterfall(
    shap_explanation,
    max_display=12
)


# In[127]:


sensitivity_exclusions = [
    "Historical_Risk_Incidents",
    "Previous_Delivery_Success_Rate",
    "Risk_Management_Maturity",
    "Seasonal_Risk_Factor"
]

X_conservative = X.drop(
    columns=sensitivity_exclusions
)

print("Original predictors:", X.shape[1])
print(
    "Conservative predictors:",
    X_conservative.shape[1]
)


# In[128]:


X_train_cons, X_test_cons, y_train_cons, y_test_cons = (
    train_test_split(
        X_conservative,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
)


# In[129]:


numeric_cons = [
    col for col in X_conservative.columns
    if is_numeric_dtype(
        X_conservative[col]
    )
]

categorical_cons = [
    col for col in X_conservative.columns
    if col not in numeric_cons
]

numeric_transformer_cons = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer_cons = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor_cons = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer_cons,
            numeric_cons
        ),
        (
            "cat",
            categorical_transformer_cons,
            categorical_cons
        )
    ]
)


# In[130]:


lr_conservative = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor_cons
        ),
        (
            "classifier",
            LogisticRegression(
                C=1,
                solver="liblinear",
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

lr_conservative.fit(
    X_train_cons,
    y_train_cons
)


# In[131]:


cons_pred = lr_conservative.predict(
    X_test_cons
)

cons_prob = lr_conservative.predict_proba(
    X_test_cons
)[:, 1]

conservative_results = {
    "Accuracy":
        round(
            accuracy_score(
                y_test_cons,
                cons_pred
            ),
            3
        ),

    "Precision":
        round(
            precision_score(
                y_test_cons,
                cons_pred
            ),
            3
        ),

    "Recall":
        round(
            recall_score(
                y_test_cons,
                cons_pred
            ),
            3
        ),

    "F1":
        round(
            f1_score(
                y_test_cons,
                cons_pred
            ),
            3
        ),

    "ROC-AUC":
        round(
            roc_auc_score(
                y_test_cons,
                cons_prob
            ),
            3
        )
}

conservative_results


# In[132]:


# Create full-dataset target
df_full = df.copy()

df_full["Elevated_Risk"] = (
    df_full["Risk_Level"]
    .isin(["High", "Critical"])
    .astype(int)
)

y_full = df_full["Elevated_Risk"]

X_full = df_full.drop(
    columns=[
        "Risk_Level",
        "Elevated_Risk",
        "Project_ID"
    ]
)

print("Full dataset X shape:", X_full.shape)
print("Full dataset y shape:", y_full.shape)
print(y_full.value_counts())


# In[133]:


numeric_full = [
    col for col in X_full.columns
    if is_numeric_dtype(X_full[col])
]

categorical_full = [
    col for col in X_full.columns
    if col not in numeric_full
]

numeric_transformer_full = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer_full = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor_full = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_full, numeric_full),
        ("cat", categorical_transformer_full, categorical_full)
    ]
)


# In[134]:


X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y_full,
    test_size=0.30,
    random_state=42,
    stratify=y_full
)


# In[135]:


lr_full = Pipeline(
    steps=[
        ("preprocessor", preprocessor_full),
        (
            "classifier",
            LogisticRegression(
                C=1,
                solver="liblinear",
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

lr_full.fit(
    X_train_full,
    y_train_full
)


# In[136]:


full_pred = lr_full.predict(
    X_test_full
)

full_prob = lr_full.predict_proba(
    X_test_full
)[:, 1]

full_scope_results = {
    "Accuracy":
        round(
            accuracy_score(
                y_test_full,
                full_pred
            ),
            3
        ),

    "Precision":
        round(
            precision_score(
                y_test_full,
                full_pred
            ),
            3
        ),

    "Recall":
        round(
            recall_score(
                y_test_full,
                full_pred
            ),
            3
        ),

    "F1":
        round(
            f1_score(
                y_test_full,
                full_pred
            ),
            3
        ),

    "ROC-AUC":
        round(
            roc_auc_score(
                y_test_full,
                full_prob
            ),
            3
        )
}

full_scope_results


# In[137]:


scope_comparison = pd.DataFrame({
    "Scope": [
        "Agile/IT Primary",
        "Full Dataset"
    ],
    "Accuracy": [
        0.913,
        full_scope_results["Accuracy"]
    ],
    "Precision": [
        0.906,
        full_scope_results["Precision"]
    ],
    "Recall": [
        0.873,
        full_scope_results["Recall"]
    ],
    "F1": [
        0.890,
        full_scope_results["F1"]
    ],
    "ROC-AUC": [
        0.970,
        full_scope_results["ROC-AUC"]
    ]
})

scope_comparison


# In[138]:


y_multi = agile_it_df["Risk_Level"]

X_multi = agile_it_df.drop(
    columns=[
        "Risk_Level",
        "Elevated_Risk",
        "Project_ID"
    ]
)

print("X_multi shape:", X_multi.shape)
print("\nClass distribution:")
print(y_multi.value_counts())


# In[139]:


X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi,
    y_multi,
    test_size=0.30,
    random_state=42,
    stratify=y_multi
)

print("Train:", X_train_multi.shape)
print("Test:", X_test_multi.shape)


# In[140]:


lr_multi = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=1,
                solver="lbfgs",
                max_iter=3000,
                random_state=42
            )
        )
    ]
)


# In[141]:


rf_multi = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features="log2",
                max_depth=None,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# In[142]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_train_multi_encoded = label_encoder.fit_transform(y_train_multi)
y_test_multi_encoded = label_encoder.transform(y_test_multi)

print(label_encoder.classes_)


# In[143]:


xgb_multi = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                subsample=0.8,
                n_estimators=500,
                max_depth=2,
                learning_rate=0.1,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss"
            )
        )
    ]
)


# In[144]:


from sklearn.metrics import accuracy_score, f1_score

multi_results = []

lr_multi.fit(X_train_multi, y_train_multi)
lr_pred_multi = lr_multi.predict(X_test_multi)

multi_results.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test_multi, lr_pred_multi),
    "Macro-F1": f1_score(y_test_multi, lr_pred_multi, average="macro"),
    "Weighted-F1": f1_score(y_test_multi, lr_pred_multi, average="weighted")
})

rf_multi.fit(X_train_multi, y_train_multi)
rf_pred_multi = rf_multi.predict(X_test_multi)

multi_results.append({
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test_multi, rf_pred_multi),
    "Macro-F1": f1_score(y_test_multi, rf_pred_multi, average="macro"),
    "Weighted-F1": f1_score(y_test_multi, rf_pred_multi, average="weighted")
})


# In[145]:


xgb_multi.fit(
    X_train_multi,
    y_train_multi_encoded
)

xgb_pred_encoded = xgb_multi.predict(
    X_test_multi
)

xgb_pred_multi = label_encoder.inverse_transform(
    xgb_pred_encoded
)

multi_results.append({
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_test_multi, xgb_pred_multi),
    "Macro-F1": f1_score(y_test_multi, xgb_pred_multi, average="macro"),
    "Weighted-F1": f1_score(y_test_multi, xgb_pred_multi, average="weighted")
})


# In[146]:


multi_results_df = pd.DataFrame(multi_results).round(3)
multi_results_df


# In[147]:


ConfusionMatrixDisplay.from_predictions(
    y_test_multi,
    lr_pred_multi,
    labels=["Low", "Medium", "High", "Critical"],
    cmap="Blues"
)

plt.title("Multiclass Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()


# In[ ]:





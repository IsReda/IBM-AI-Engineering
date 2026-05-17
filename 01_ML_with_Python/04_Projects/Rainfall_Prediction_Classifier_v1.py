# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🌧️ Rainfall Prediction Classifier — Melbourne, Australia
#
# > 🧠 An end-to-end supervised learning project where I build, tune and compare two classifiers (Random Forest 🌲 and Logistic Regression 📉) that predict whether it will rain *today* in the Melbourne metropolitan area using historical weather observations.
#
# ---
#
# ## 📋 Overview
#
# In this notebook I work through a complete machine-learning workflow on the **Australian Weather** dataset (Bureau of Meteorology, 2008–2017). I focus on the Melbourne / Melbourne Airport / Watsonia cluster to keep the problem locally meaningful.
#
# ### 🗂️ Table of Contents
#
# | Section | Title |
# |---|---|
# | 🧩 | **Theory & Notation** |
# | ⚙️ | **Part 1 — Environment & Imports** |
# | 📥 | **Part 2 — Loading the Data** |
# | 🧹 | **Part 3 — Cleaning & Feature Engineering** |
# | 🔢 | **Part 4 — Class Balance & Target Definition** |
# | ✂️ | **Part 5 — Train / Test Split with Stratification** |
# | 🏗️ | **Part 6 — Preprocessing Pipeline** |
# | 🌲 | **Part 7 — Random Forest + Grid Search** |
# | 🎯 | **Part 8 — Evaluation: Classification Report & Confusion Matrix** |
# | 🔍 | **Part 9 — Feature Importances** |
# | 📉 | **Part 10 — Swap-in Logistic Regression** |
# | 📊 | **Summary** |
# | 🧪 | **Sandbox** |
#

# %% [markdown]
# ## 🧩 Theory & Notation
#
# ### 🎯 The classification problem
#
# I am given a feature matrix $X \in \mathbb{R}^{n \times d}$ and a binary target $y \in \{0, 1\}^n$ where
#
# $$
# y_i \;=\; \begin{cases} 1 & \text{if it rains on day } i \\ 0 & \text{otherwise} \end{cases}
# $$
#
# The goal is to learn a function $f_\theta : \mathbb{R}^d \to \{0, 1\}$ that minimises the expected misclassification risk
#
# $$
# \mathcal{R}(f_\theta) \;=\; \mathbb{E}_{(X, y)}\bigl[\mathbb{1}\{f_\theta(X) \neq y\}\bigr].
# $$
#
# ### 🌲 Random Forest
#
# A Random Forest aggregates $B$ decision trees $\{T_b\}_{b=1}^{B}$ each trained on a bootstrap sample. The forest predicts by majority vote:
#
# $$
# \hat{y} \;=\; \operatorname*{arg\,max}_{c \in \{0,1\}} \sum_{b=1}^{B} \mathbb{1}\{T_b(x) = c\}.
# $$
#
# Each split inside a tree chooses the feature/threshold that maximises information gain measured by Gini impurity:
#
# $$
# \text{Gini}(S) \;=\; 1 - \sum_{c \in \{0,1\}} p_c^2, \qquad p_c = \frac{|\{i \in S : y_i = c\}|}{|S|}.
# $$
#
# ### 📉 Logistic Regression
#
# Logistic Regression models the log-odds linearly:
#
# $$
# \log\frac{P(y=1 \mid x)}{P(y=0 \mid x)} \;=\; w^\top x + b, \qquad P(y=1 \mid x) \;=\; \sigma(w^\top x + b) \;=\; \frac{1}{1 + e^{-(w^\top x + b)}}.
# $$
#
# Training minimises the regularised cross-entropy loss
#
# $$
# \mathcal{L}(w, b) \;=\; -\frac{1}{n}\sum_{i=1}^{n}\bigl[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\bigr] + \lambda \|w\|_p^p,
# $$
#
# where $p=1$ gives L1 (Lasso) and $p=2$ gives L2 (Ridge) regularisation.
#
# ### 📈 Evaluation metrics
#
# Given the confusion matrix entries $\text{TP}, \text{TN}, \text{FP}, \text{FN}$:
#
# | Metric | Formula | Intuition |
# |---|---|---|
# | Accuracy | $\dfrac{TP + TN}{TP + TN + FP + FN}$ | overall fraction correct |
# | Precision | $\dfrac{TP}{TP + FP}$ | of predicted-rain days, how many actually rained |
# | Recall (TPR) | $\dfrac{TP}{TP + FN}$ | of actually-rainy days, how many I caught |
# | F1 | $2\cdot\dfrac{\text{Prec}\cdot\text{Rec}}{\text{Prec} + \text{Rec}}$ | harmonic mean |
#
# ### 🔢 The dataset schema
#
# | Field | Description | Unit | Type |
# | :--- | :--- | :--- | :--- |
# | Date | Date of observation (YYYY-MM-DD) | Date | object |
# | Location | Weather station location | — | object |
# | MinTemp / MaxTemp | Min/max temperature | °C | float |
# | Rainfall | Total daily rainfall | mm | float |
# | Evaporation | Daily evaporation | mm | float |
# | Sunshine | Bright sunshine | hours | float |
# | WindGustDir / WindGustSpeed | Strongest gust direction & speed | compass / km·h⁻¹ | object/float |
# | WindDir9am / WindDir3pm | Wind direction at 9am / 3pm | compass | object |
# | WindSpeed9am / WindSpeed3pm | Wind speed at 9am / 3pm | km·h⁻¹ | float |
# | Humidity9am / Humidity3pm | Relative humidity at 9am / 3pm | % | float |
# | Pressure9am / Pressure3pm | MSL pressure at 9am / 3pm | hPa | float |
# | Cloud9am / Cloud3pm | Cloud cover at 9am / 3pm | eighths | float |
# | Temp9am / Temp3pm | Temperature at 9am / 3pm | °C | float |
# | RainToday | At least 1 mm rain today? | Yes/No | object |
# | RainTomorrow | At least 1 mm rain tomorrow? | Yes/No | object |
#

# %% [markdown]
# ## ⚙️ Part 1 — Environment & Imports
#
# First I install and import everything I will need. I use:
#
# - **pandas** for tabular data 🐼
# - **matplotlib** & **seaborn** for visualisation 📈
# - **scikit-learn** for pipelines, transformers, the two classifiers, grid search and metrics 🧠
#
# I prefer composing everything inside a `Pipeline` + `ColumnTransformer` so the preprocessing is **fit only on training folds** during cross-validation — this prevents subtle data leakage.

# %%
# 📦 Install the required libraries (uncomment if running fresh)
# # !pip install numpy
# # !pip install pandas
# # !pip install matplotlib
# # !pip install scikit-learn
# # !pip install seaborn

# %%
# ⚙️ Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# %% [markdown]
# ## 📥 Part 2 — Loading the Data
#
# I load the CSV directly from the public bucket. The dataset is a daily-frequency time series across many Australian weather stations.

# %%
# 📥 Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

# %% [markdown]
# ### 🔍 Inspecting missingness
#
# A quick `.count()` per column tells me how many non-null values each feature has. Sunshine and cloud-cover features are visually appealing predictors but suffer from heavy missingness — too much to impute reliably.

# %%
df.count()

# %% [markdown]
# ### 🧹 Drop rows with missing values
#
# To keep the pipeline simple and avoid imputation bias I drop incomplete rows. After dropping I still have tens of thousands of observations, so the loss is acceptable.

# %%
df = df.dropna()
df.info()

# %%
df.columns

# %% [markdown]
# ## 🧹 Part 3 — Cleaning & Feature Engineering
#
# ### ⚠️ Data leakage considerations
#
# Several features (e.g. `Rainfall`, `Sunshine`, `Cloud3pm`) are recorded **at the end** of the observation day. If my target is *tomorrow's* rain, using such same-day measurements is fine — but the original `RainTomorrow` column technically reflects an event we won't have observed yet. To make the problem realistic and reusable, I **reframe the task to predict today's rain** from yesterday's recorded weather. This is the practical "should I bike to work today?" framing.
#
# ### 📝 Points to note — 1: features that are inefficient for predicting tomorrow's rain
#
# - **Date & Location (raw):** while they provide context, using them directly without transformation (extracting months, encoding stations, …) makes the model overfit specific years/stations instead of learning general patterns.
# - **Leakage risk:** any column recorded after the rain started "tomorrow" cannot be used in a real-world forecast.
# - **Inefficient features:** some metrics correlate weakly with next-day rainfall or duplicate information already present in more precise atmospheric measurements (e.g. `Cloud9am` vs `Humidity3pm`).
#
# > 💡 Practical tip: features evaluated over the *entire* duration of the prediction day (full-day rainfall, total sunshine hours) are exactly the leakage suspects.
#
# ### 🔄 Rename the rain columns to reflect the new target
#
# Since I will predict **today's** rain from **yesterday's** weather, I rename:
#
# - `RainToday` → `RainYesterday` (lagged predictor)
# - `RainTomorrow` → `RainToday` (target) 🎯

# %%
# 🔄 Rename to match the new framing
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'})
print(df.columns)

# %% [markdown]
# ### 🗺️ Data granularity — narrowing to one region
#
# Australia is huge and rainfall patterns vary enormously by location. Training a single model on every station forces it to fit many local regimes at once. I focus on a small geographic cluster instead.
#
# ### 📍 Location selection: Melbourne metropolitan cluster
#
# - **Watsonia** is only ~15 km from Melbourne.
# - **Melbourne Airport** is only ~18 km from Melbourne.
#
# Grouping these three stations gives me a coherent local dataset. I keep `Location` as a categorical variable to let the model exploit small inter-station differences.

# %%
# 🗺️ Restrict to the Melbourne cluster
df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]
df.info()


# %% [markdown]
# I retain roughly **7,557** records — enough to fit a non-trivial model. If I needed more, I could expand the cluster (e.g. include Ballarat) or pull a longer time range from the source.
#
# ### 🌱 Engineering a seasonality feature
#
# Weather is strongly seasonal. Rather than passing the raw `Date` to the model, I derive a categorical **`Season`** feature, which is more compact and more directly informative. Australia is in the southern hemisphere, so the season ↔ month mapping is shifted by six months relative to the northern hemisphere:
#
# | Months | Season (Southern Hemisphere) |
# |---|---|
# | Dec / Jan / Feb | Summer ☀️ |
# | Mar / Apr / May | Autumn 🍂 |
# | Jun / Jul / Aug | Winter ❄️ |
# | Sep / Oct / Nov | Spring 🌸 |
#

# %%
# 🌱 Map a date to its Southern-Hemisphere season
def date_to_season(date):
    month = date.month
    if month in (12, 1, 2):
        return 'Summer'
    elif month in (3, 4, 5):
        return 'Autumn'
    elif month in (6, 7, 8):
        return 'Winter'
    elif month in (9, 10, 11):
        return 'Spring'


# %% [markdown]
# ### ✅ Exercise 1 — Apply the season mapping and drop `Date`
#
# I parse the `Date` column into datetime, apply my mapping, and drop the raw `Date` column since `Season` carries the relevant signal.

# %%
# ✅ Exercise 1 — convert Date → Season and drop Date
# 1. Parse the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# 2. Apply the mapping to create the 'Season' feature
df['Season'] = df['Date'].apply(date_to_season)

# 3. Drop the now-redundant 'Date' column
df = df.drop(columns=['Date'])
df.head()

# %% [markdown]
# Good — the feature set is ready. Before fitting any model I want to check the **class balance** of the target, because imbalance affects both metric choice and model behaviour.

# %% [markdown]
# ## 🔢 Part 4 — Class Balance & Target Definition
#
# ### ✅ Exercise 2 — Define features `X` and target `y`
#
# I separate the predictors from the target. The target is `RainToday` (recall: this is the *renamed* column representing the day I want to predict).

# %%
# ✅ Exercise 2 — define X and y
X = df.drop('RainToday', axis=1)
y = df['RainToday']
print(X.columns)
print(y.name)

# %% [markdown]
# ### ✅ Exercise 3 — How balanced are the classes?

# %%
# ✅ Exercise 3 — class counts
y.value_counts()

# %% [markdown]
# ### 📝 Exercise 4 — What do the counts tell me?
#
# Counts: **5,766 "No" / 1,791 "Yes"** → roughly a **3 : 1** imbalance.
#
# | Question | My answer |
# |---|---|
# | How often does it rain in Melbourne? | ≈ **23.7 %** of days (≈ 1 in 4). |
# | Baseline accuracy if I always predict "No rain"? | ≈ **76.3 %**. Any useful model must beat this. |
# | Is the dataset balanced? | **No** — imbalanced 3 : 1 toward "No". |
# | Next steps? | (1) Encode categoricals & scale numerics; (2) Stratified train/test split; (3) Use **F1 / recall / confusion matrix**, not just accuracy. |
#
# > ⚠️ With this kind of imbalance, "accuracy" can be a misleading headline metric. A model that *never* predicts rain still scores 76 % — useless for someone deciding whether to take an umbrella. I will pay particular attention to the **recall on the "Yes" class**.
#

# %% [markdown]
# ## ✂️ Part 5 — Train / Test Split with Stratification
#
# ### ✅ Exercise 5 — Split with target stratification
#
# Because the target is imbalanced I pass `stratify=y` so both the training and test sets keep the same ≈ 24 % positive rate. I reserve 20 % for testing and fix `random_state=42` for reproducibility.

# %%
# ✅ Exercise 5 — stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# %% [markdown]
# ## 🏗️ Part 6 — Preprocessing Pipeline
#
# I now build a `ColumnTransformer` that applies different preprocessing to numeric vs categorical columns, and wrap it inside a `Pipeline` together with the classifier. This means the preprocessing is **re-fitted on every CV fold** — no leakage from validation into training.
#
# ### ✅ Exercise 6 — Auto-detect numeric and categorical columns
#
# I let pandas dtypes drive feature-type detection. `number` covers `int*` and `float*`; `object` and `category` cover the string categoricals.

# %%
# ✅ Exercise 6 — detect column types automatically
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print("🔢 Numeric features  :", numeric_features)
print("🔠 Categorical features:", categorical_features)

# %% [markdown]
# ### ⚙️ Define the two transformers
#
# - **Numeric** → `StandardScaler` standardises each feature to zero mean / unit variance:
#   $$x' = \frac{x - \mu}{\sigma}.$$
#   Crucial for Logistic Regression (the gradient is poorly conditioned otherwise). Trees don't strictly need it but it does no harm.
# - **Categorical** → `OneHotEncoder` with `handle_unknown='ignore'` so unseen categories in the test set don't raise an error.

# %%
# ⚙️ Scale numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# ⚙️ One-hot encode categorical features
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# %% [markdown]
# ### ✅ Exercise 7 — Combine the transformers into a `ColumnTransformer`

# %%
# ✅ Exercise 7 — combine into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# %% [markdown]
# ### ✅ Exercise 8 — Wrap preprocessor + Random Forest into one pipeline
#
# I stack the preprocessor with a `RandomForestClassifier` so the whole thing behaves as a single estimator. This is what makes `GridSearchCV` ergonomic later.

# %%
# ✅ Exercise 8 — full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# %% [markdown]
# ## 🌲 Part 7 — Random Forest + Grid Search Cross-Validation
#
# ### 🔍 Hyperparameter grid
#
# I sweep three key Random Forest hyperparameters:
#
# | Parameter | Meaning | Values |
# |---|---|---|
# | `n_estimators` | Number of trees $B$ | 50, 100 |
# | `max_depth` | Max depth of each tree | None, 10, 20 |
# | `min_samples_split` | Min samples to split a node | 2, 5 |
#
# That gives $2 \times 3 \times 2 = 12$ configurations, each evaluated by 5-fold cross-validation → $60$ fits in total.

# %%
# 🔍 Hyperparameter grid for Random Forest
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# %% [markdown]
# ### 🔄 Pipeline reuse inside cross-validation
#
# Inside the grid search, the **same pipeline** is fit on each inner training fold and predicted on the corresponding validation fold. Because the scaler and one-hot encoder live *inside* the pipeline, their statistics (means, std-devs, category sets) are computed only on the training fold — so no validation data leaks into the preprocessing step.
#
# ### ⚙️ Cross-validation scheme
#
# I use **`StratifiedKFold` with 5 splits** to preserve the rain/no-rain ratio in each fold. This matters more here than usual because of the 3 : 1 imbalance.

# %%
# ⚙️ Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True)

# %% [markdown]
# ### ✅ Exercise 9 — Fit `GridSearchCV`
#
# I optimise for `accuracy` (matching the course objective). `verbose=2` prints each fit so I can watch progress.

# %%
# ✅ Exercise 9 — instantiate and fit GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)
grid_search.fit(X_train, y_train)

# %% [markdown]
# ### 🏆 Best parameters and best CV score

# %%
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# %% [markdown]
# ### ✅ Exercise 10 — Test-set accuracy
#
# The CV score estimates generalisation, but the held-out test set is the unbiased final check.

# %%
# ✅ Exercise 10 — score on the test set
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# %% [markdown]
# I obtain a Random Forest that classifies Melbourne-area rainfall with **≈ 84 % accuracy** on unseen data. That sounds good — but with a 76 % "always say No" baseline, I need a deeper look. Let's go beyond accuracy.

# %% [markdown]
# ## 🎯 Part 8 — Evaluation: Classification Report & Confusion Matrix
#
# The best estimator is stored on the `grid_search` object — calling `.predict()` on it implicitly runs the full pipeline (preprocess → classify) on new data.
#
# ### ✅ Exercise 11 — Predict on the unseen test set

# %%
# ✅ Exercise 11 — predictions on the test set
y_pred = grid_search.predict(X_test)

# %% [markdown]
# ### ✅ Exercise 12 — Classification report
#
# The classification report shows **precision, recall and F1** for each class plus weighted averages. For an imbalanced problem these per-class numbers are far more informative than overall accuracy.

# %%
# ✅ Exercise 12 — full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### ✅ Exercise 13 — Confusion matrix
#
# The confusion matrix lays out the four prediction outcomes:
#
# $$
# \mathbf{C} \;=\; \begin{pmatrix} TN & FP \\ FN & TP \end{pmatrix}
# $$
#
# Visualising it makes the **false-negative rate** (rainy days I missed) immediately obvious.

# %%
# ✅ Exercise 13 — confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — Random Forest')
plt.show()

# %% [markdown]
# ### 📝 Points to note — 2: the True Positive Rate
#
# Reading off the confusion matrix:
#
# $$
# \text{TPR} \;=\; \frac{TP}{TP + FN} \;=\; \frac{178}{178 + 180} \;\approx\; 0.497 \;=\; \mathbf{49.7\%}.
# $$
#
# So even though the headline accuracy is ~84 %, the model **catches only about half of the actual rainy days**. The high accuracy is buoyed by the dominant "No rain" class. This is exactly the trap I was wary of with imbalanced data — and it motivates trying a different classifier or a class-weighting scheme.
#

# %% [markdown]
# ## 🔍 Part 9 — Feature Importances
#
# Random Forests provide a built-in importance score per *transformed* feature (Mean Decrease in Impurity). To make the chart human-readable I need to map the one-hot-encoded names back to my original features.
#
# ### ✅ Exercise 14 — Extract the importances

# %%
# ✅ Exercise 14 — pull importances off the fitted classifier
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# %% [markdown]
# ### 📊 Build a tidy importance dataframe and plot the top 20
#
# `numeric_features` are passed through unchanged, but every categorical column expands into one column per category after one-hot encoding. I rebuild the full ordered feature-name list by concatenating numerics with the encoder's `get_feature_names_out()`.

# %%
# 📊 Rebuild the full feature-name list (numerics + expanded categoricals)
feature_names = numeric_features + list(
    grid_search.best_estimator_['preprocessor']
               .named_transformers_['cat']
               .named_steps['onehot']
               .get_feature_names_out(categorical_features)
)

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

N = 20  # 🔢 tweak to display more or fewer features
top_features = importance_df.head(N)

# 📈 Plot
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # most important at the top
plt.title(f'Top {N} Most Important Features for Predicting Rain Today')
plt.xlabel('Importance Score')
plt.show()

# %% [markdown]
# ### 📝 Point to note — 3: most important feature
#
# From the bar chart, the dominant predictor is **`Humidity3pm`** — by a clear margin. This is intuitive: high mid-afternoon humidity is a strong indicator of impending precipitation. Atmospheric pressure (`Pressure3pm`, `Pressure9am`) and afternoon cloud cover follow.
#

# %% [markdown]
# ## 📉 Part 10 — Swap-in: Logistic Regression
#
# ### 💡 Why swap models at all?
#
# In practice I want to compare a strong non-linear model (Random Forest) with a strong **linear** baseline (Logistic Regression). Linear models are easier to interpret, faster to train, and — with appropriate class weights — sometimes catch the minority class better.
#
# Because everything lives inside a single `Pipeline`, switching the classifier is a one-liner: `pipeline.set_params(classifier=LogisticRegression(...))`. The preprocessing stays identical.
#
# ### ✅ Exercise 15 — Update the pipeline and the parameter grid
#
# For Logistic Regression I sweep:
#
# | Parameter | Values | Effect |
# |---|---|---|
# | `solver` | `'liblinear'` | supports both L1 and L2 |
# | `penalty` | `'l1'`, `'l2'` | sparsity vs. shrinkage |
# | `class_weight` | `None`, `'balanced'` | downweights the majority class to fight imbalance |

# %%
# ✅ Exercise 15 — swap classifier and update grid
# 🔄 Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# 🔄 Update the grid search to use the new pipeline
grid_search.estimator = pipeline

# 🆕 New parameter grid for Logistic Regression
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

grid_search.param_grid = param_grid

# 🚀 Refit on the training data
grid_search.fit(X_train, y_train)

# 🎯 Predict on the held-out test set
y_pred = grid_search.predict(X_test)

# %% [markdown]
# ### 📊 Compare to the previous model
#
# I print the classification report and draw the confusion matrix with seaborn for variety.

# %%
# 📊 Classification report
print(classification_report(y_test, y_pred))

# 📊 Confusion matrix as a heatmap
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix — Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 📝 Points to note — 4: comparing the two models
#
# 1. **Accuracy.** The Random Forest reaches ≈ **84 %** overall accuracy; Logistic Regression sits slightly below at ≈ **83 %**.
# 2. **True positives (rain correctly caught).** Logistic Regression catches **182** rainy days — slightly more than Random Forest's **178**.
# 3. **True positive rate (recall on rain).** Random Forest TPR ≈ **49.7 %**; Logistic Regression TPR ≈ **50.8 %** — marginally better at the imbalanced class.
# 4. **False positives (false alarms).** Logistic Regression triggers **84** false alarms vs. Random Forest's **57** — a noticeable jump. Logistic Regression sacrifices some precision to gain a bit of recall.
#
# > 🎯 **Takeaway.** If the cost of a missed rainy day (forgotten umbrella) is higher than the cost of a false alarm (carrying an unused umbrella), the LR model is preferable. If we want fewer false alarms at the expense of slightly worse recall, RF wins. Both beat the 76 % "always-No" baseline, so both have learned something — but neither is great at the minority class. Real improvements would likely come from class-imbalance handling (SMOTE, threshold tuning), better features, or boosted-tree alternatives (XGBoost, LightGBM).
#

# %% [markdown]
# ## 📊 Summary
#
# ### What I built
#
# | Stage | What I did | Why it mattered |
# |---|---|---|
# | 📥 **Data load** | Read Australia weather CSV, dropped NaNs | Avoid imputation bias on a still-large dataset |
# | 🧹 **Reframe target** | Renamed `RainTomorrow` → `RainToday` | Avoid leakage; practical "should I bike today?" framing |
# | 🗺️ **Region filter** | Melbourne + Melb. Airport + Watsonia | Single local weather regime → cleaner learning signal |
# | 🌱 **Feature engineering** | `Date` → `Season` (S-hemisphere mapping) | Compact seasonal feature beats raw timestamps |
# | ✂️ **Split** | 80 / 20 with `stratify=y`, `random_state=42` | Preserve 24 % positive rate in both sets |
# | 🏗️ **Pipeline** | `StandardScaler` on numerics, `OneHotEncoder` on categoricals, all in `ColumnTransformer` | No leakage during CV; reproducible |
# | 🌲 **Model A** | Random Forest + 5-fold stratified grid search on $n\_estimators$ / $max\_depth$ / $min\_samples\_split$ | Strong non-linear baseline |
# | 📉 **Model B** | Logistic Regression with `penalty` / `class_weight` sweep | Interpretable linear baseline + class-imbalance lever |
# | 🎯 **Evaluation** | Classification report + confusion matrix + feature importances | Beyond raw accuracy; understand the imbalance trade-off |
#
# ### Final model comparison
#
# | Metric | Random Forest 🌲 | Logistic Regression 📉 | Baseline (always-No) |
# |---|---|---|---|
# | Accuracy | **≈ 0.84** | ≈ 0.83 | 0.76 |
# | TPR (recall, "Yes") | 49.7 % | **50.8 %** | 0 % |
# | TP count | 178 | **182** | 0 |
# | FP count | **57** | 84 | 0 |
# | FN count | 180 | **176** | 358 |
# | Most important feature | **`Humidity3pm`** | — | — |
#
# ### Key lessons
#
# - ⚠️ Accuracy alone is a misleading metric on imbalanced data; always inspect per-class recall.
# - 🏗️ Wrapping preprocessing inside a `Pipeline` is the cleanest way to prevent CV leakage.
# - 🔄 Swapping classifiers downstream of a shared preprocessing pipeline is a one-liner — this is the real super-power of `sklearn.pipeline.Pipeline`.
# - 🌱 Domain-aware features (season, region cluster) often matter more than model choice.
# - 🎯 No model in this notebook truly cracks the rainy-day class — that's the natural next investigation (resampling, threshold tuning, gradient boosting).
#

# %% [markdown]
# ## 🧪 Sandbox — Free Experimentation
#
# This section is for me to extend the project further. Some ideas worth trying:
#
# 1. 🌡️ **Threshold tuning** — pull `predict_proba`, sweep the classification threshold, plot the precision-recall curve, and pick the threshold that maximises F1 on the validation fold instead of using the default 0.5.
# 2. ⚖️ **Resampling** — apply SMOTE or random oversampling on the training set to balance the classes, then re-evaluate.
# 3. 🌲 **Gradient boosting** — swap in `HistGradientBoostingClassifier` (or XGBoost / LightGBM) which usually outperforms Random Forest on tabular data.
# 4. 🗺️ **Region expansion** — add another nearby station cluster (e.g. Ballarat / Bendigo) and let `Location` carry the regional signal.
# 5. 📅 **Richer time features** — `Month`, `DayOfYear`, sine/cosine seasonal encodings.
# 6. 🧪 **Permutation importance** — replace MDI importances (biased toward high-cardinality numerics) with `sklearn.inspection.permutation_importance`.
# 7. 📉 **Calibration** — check whether predicted probabilities are well-calibrated with `CalibratedClassifierCV` and a reliability diagram.

# %%
# 🧪 Sandbox cell 1 — threshold tuning starter
# from sklearn.metrics import precision_recall_curve
# y_proba = grid_search.predict_proba(X_test)[:, list(grid_search.classes_).index('Yes')]
# precision, recall, thresholds = precision_recall_curve(
#     (y_test == 'Yes').astype(int), y_proba
# )
# plt.plot(thresholds, precision[:-1], label='Precision')
# plt.plot(thresholds, recall[:-1], label='Recall')
# plt.xlabel('Threshold'); plt.ylabel('Score'); plt.legend(); plt.grid(); plt.show()


# %%
# 🧪 Sandbox cell 2 — try gradient boosting
# from sklearn.ensemble import HistGradientBoostingClassifier
# pipeline.set_params(classifier=HistGradientBoostingClassifier(random_state=42))
# grid_search.estimator = pipeline
# grid_search.param_grid = {
#     'classifier__learning_rate': [0.05, 0.1],
#     'classifier__max_depth':     [None, 6, 10],
#     'classifier__max_iter':      [100, 200],
# }
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_, grid_search.best_score_)


# %%
# 🧪 Sandbox cell 3 — permutation importance
# from sklearn.inspection import permutation_importance
# r = permutation_importance(grid_search.best_estimator_, X_test, y_test,
#                            n_repeats=10, random_state=42, n_jobs=-1)
# perm = pd.DataFrame({'Feature': X_test.columns, 'Importance': r.importances_mean}) \
#         .sort_values('Importance', ascending=False)
# print(perm.head(15))


# %%
# 🧪 Sandbox cell 4 — free space for my own experiments


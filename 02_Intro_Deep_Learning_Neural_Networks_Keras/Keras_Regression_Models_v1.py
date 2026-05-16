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
# # 🧠 Regression Models with Keras
#
# > **Course:** IBM AI Engineering Professional Certificate — Module 02: Deep Learning with Keras  
# > **Topic:** Building, Training & Evaluating Neural Network Regression Models  
# > **Framework:** Keras (TensorFlow backend)
#

# %% [markdown]
# ## 📋 Overview
#
# In this notebook I build a deep learning **regression model** using the Keras high-level API to predict concrete compressive strength from its mix ingredients. Keras abstracts the low-level complexity of TensorFlow, letting me define, compile, and train networks in just a few lines — making it the ideal starting point for applied deep learning.
#
# ### 📚 Table of Contents
# | # | Section | Description |
# |---|---------|-------------|
# | 1 | 🔢 Data Acquisition & Cleaning | Load the concrete dataset, inspect & normalise |
# | 2 | ⚙️ Keras Setup | Import Sequential API, Dense & Input layers |
# | 3 | 🏗️ Neural Network Architecture | Define the regression model function |
# | 4 | 📈 Training & Evaluation | Fit, validate, and analyse training behaviour |
# | 5 | 🧪 Extended Architecture — 5 Hidden Layers | Deeper network experiment |
# | 6 | 📊 Summary | Comparative summary table |
# | 7 | 🧪 Sandbox | Free experimentation zone |
#

# %% [markdown]
# ## 🧩 Theory
#
# ### 🧠 What Is a Neural Network Regression Model?
#
# A **regression neural network** maps a vector of input features $\mathbf{x} \in \mathbb{R}^n$ to a continuous scalar output $\hat{y} \in \mathbb{R}$. Unlike classification, the output layer has **no activation function** (linear activation), allowing it to predict any real value.
#
# ---
#
# ### ➕ Forward Pass — Weighted Sum per Neuron
#
# For each neuron $j$ in a hidden layer, the **pre-activation** (weighted sum) is:
#
# $$z_j = \sum_{i=1}^{n} w_{ij}\, x_i + b_j$$
#
# where $w_{ij}$ are learnable weights and $b_j$ is the bias term.
#
# ---
#
# ### 📈 Activation Function — ReLU
#
# The **Rectified Linear Unit (ReLU)** introduces non-linearity so the network can model complex relationships:
#
# $$\text{ReLU}(z) = \max(0,\, z)$$
#
# It avoids the vanishing-gradient problem common with sigmoid/tanh and is computationally efficient.
#
# ---
#
# ### 🎯 Loss Function — Mean Squared Error (MSE)
#
# For regression, I minimise the **Mean Squared Error** over $m$ training samples:
#
# $$\mathcal{L}_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^{m} \left( y_i - \hat{y}_i \right)^2$$
#
# MSE penalises large errors heavily (quadratic penalty), making it sensitive to outliers but well-suited for continuous targets.
#
# ---
#
# ### 🔄 Backpropagation & Weight Update (Adam Optimiser)
#
# The **Adam (Adaptive Moment Estimation)** optimiser computes adaptive learning rates for each parameter using first and second moment estimates of the gradients:
#
# $$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t$$
# $$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2$$
# $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
# $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\, \hat{m}_t$$
#
# Default hyperparameters: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
#
# ---
#
# ### 📥 Dataset — Concrete Compressive Strength
#
# | Feature | Description | Unit |
# |---------|-------------|------|
# | Cement | Cement content | kg/m³ |
# | Blast Furnace Slag | Industrial by-product | kg/m³ |
# | Fly Ash | Coal combustion residual | kg/m³ |
# | Water | Water content | kg/m³ |
# | Superplasticizer | Chemical additive | kg/m³ |
# | Coarse Aggregate | Gravel content | kg/m³ |
# | Fine Aggregate | Sand content | kg/m³ |
# | Age | Age of the concrete sample | Days |
# | **Strength** (target $y$) | **Compressive strength** | **MPa** |
#
# ---
#
# ### 🔢 Feature Normalisation (Z-score Standardisation)
#
# Raw features span very different ranges. I normalise each feature $x_i$ to zero mean and unit variance:
#
# $$x_i^{\text{norm}} = \frac{x_i - \mu_i}{\sigma_i}$$
#
# This prevents large-magnitude features from dominating gradient updates and accelerates convergence.
#

# %% [markdown]
# ## Part 1 — 🔢 Data Acquisition & Cleaning
#
# I start by installing the required libraries at specific versions for reproducibility, then suppressing TensorFlow's CPU-architecture warnings to keep output clean.
#

# %%
# ⚙️ Install pinned library versions for reproducibility
# !pip install numpy==2.0.2 --quiet
# !pip install pandas==2.2.2 --quiet
# !pip install tensorflow_cpu==2.18.0 --quiet

# %% [markdown]
# I suppress TensorFlow's oneDNN and log-level warnings — these only appear because I am running on CPU. I would comment these lines out when running on GPU hardware.
#

# %%
import os

# ⚙️ Suppress TensorFlow CPU warnings (comment out when using GPU)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
import pandas as pd
import numpy as np
import keras
import warnings

warnings.simplefilter('ignore', FutureWarning)

print(f"✅ pandas  {pd.__version__}")
print(f"✅ numpy   {np.__version__}")
print(f"✅ keras   {keras.__version__}")

# %% [markdown]
# ### 📥 Loading the Concrete Dataset
#
# I load the UCI Concrete Compressive Strength dataset directly from the remote CSV. Each row represents one concrete mix sample with 8 ingredient/age features and one continuous target: compressive strength in MPa.
#

# %%
# 📥 Load the concrete compressive strength dataset
filepath = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

print(f"🔢 Dataset shape: {concrete_data.shape}")
concrete_data.head()

# %% [markdown]
# ### 🔢 Exploratory Data Check
#
# I check the dataset shape, descriptive statistics, and missing value counts. With ~1,000 samples and 8 features, I need to be mindful of overfitting — this is a small dataset by deep learning standards.
#

# %%
# 📊 Dataset dimensions
print(f"📊 Rows: {concrete_data.shape[0]}  |  Columns: {concrete_data.shape[1]}")

# %%
# 📊 Descriptive statistics — check ranges, means, std devs
concrete_data.describe()

# %%
# ✅ Check for missing values — none expected
missing = concrete_data.isnull().sum()
print("Missing values per column:")
print(missing)
print(f"\n✅ Total missing cells: {missing.sum()}")

# %% [markdown]
# The data is clean with no missing values. Notice the wide variation in feature scales (e.g., Cement ranges ~100–540, Superplasticizer 0–32) — this makes normalisation essential.
#
# ---
#
# ### 📥 📤 Splitting Predictors and Target
#
# I separate the feature matrix $\mathbf{X}$ (all columns except `Strength`) from the target vector $\mathbf{y}$ (`Strength`). This follows the standard supervised learning setup:
#
# $$\mathbf{X} \in \mathbb{R}^{m \times 8}, \quad \mathbf{y} \in \mathbb{R}^{m}$$
#

# %%
# 📥 Feature matrix — all columns except the target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]

# 📤 Target vector — compressive strength in MPa
target = concrete_data['Strength']

print(f"📥 Predictors shape : {predictors.shape}")
print(f"📤 Target shape     : {target.shape}")

# %%
# 🔍 Sanity check — first 5 rows of predictors
predictors.head()

# %%
# 🔍 Sanity check — first 5 values of target (MPa)
target.head()

# %% [markdown]
# ### 🔢 Z-Score Feature Normalisation
#
# I apply z-score standardisation so that each feature has mean $\mu = 0$ and standard deviation $\sigma = 1$:
#
# $$x_i^{\text{norm}} = \frac{x_i - \mu_i}{\sigma_i}$$
#
# This is critical for gradient-based optimisation — without it, features with large magnitudes (e.g., Coarse Aggregate ~1000 kg/m³) would dominate the weight updates and slow convergence.
#

# %%
# 🔢 Normalise features: zero mean, unit variance
predictors_norm = (predictors - predictors.mean()) / predictors.std()

print("✅ Normalised feature statistics (should be ~0 mean, ~1 std):")
predictors_norm.describe().loc[['mean', 'std']].round(4)

# %%
# ⚙️ Save the number of input features — needed to define Input layer shape
n_cols = predictors_norm.shape[1]
print(f"📥 Number of input features (n_cols): {n_cols}")

# %% [markdown]
# ## Part 2 — ⚙️ Keras Setup
#
# I import the components I need from Keras. The **Sequential** API lets me stack layers linearly — ideal for straightforward feedforward architectures like this regression network.
#
# | Import | Role |
# |--------|------|
# | `Sequential` | Container that holds an ordered stack of layers |
# | `Dense` | Fully connected layer: $\mathbf{a} = \phi(\mathbf{W}\mathbf{x} + \mathbf{b})$ |
# | `Input` | Specifies the shape of the input tensor (8 features) |
#

# %%
from keras.models import Sequential
from keras.layers import Dense, Input

print("✅ Keras imports loaded successfully")


# %% [markdown]
# ## Part 3 — 🏗️ Neural Network Architecture
#
# ### 🧠 Model Design: 2-Hidden-Layer Regression Network
#
# I define a factory function `regression_model()` that builds and compiles the network. Encapsulating model creation in a function means I can call it repeatedly to get fresh, randomly initialised models — essential for fair experiments.
#
# **Architecture diagram:**
#
# ```
# 📥 Input  →  [8 features]
#               ↓
# 🧠 Dense(50, ReLU)   ← Hidden Layer 1
#               ↓
# 🧠 Dense(50, ReLU)   ← Hidden Layer 2
#               ↓
# 🎯 Dense(1, linear)  ← Output: predicted strength (MPa)
# ```
#
# The output layer uses **linear activation** (no activation function), which is standard for regression — the network outputs any real-valued prediction.
#
# **Compiled with:**
# - Optimiser: `adam` (adaptive learning rate)
# - Loss: `mean_squared_error` (penalises large prediction errors quadratically)
#

# %%
def regression_model():
    """
    🏗️ Build and compile a 2-hidden-layer regression neural network.
    
    Architecture:
        Input(n_cols) → Dense(50, ReLU) → Dense(50, ReLU) → Dense(1, linear)
    
    Returns:
        model: Compiled Keras Sequential model
    """
    model = Sequential()
    
    # 📥 Input layer — define the feature dimensionality
    model.add(Input(shape=(n_cols,)))
    
    # 🧠 Hidden Layer 1 — 50 neurons, ReLU activation
    model.add(Dense(50, activation='relu'))
    
    # 🧠 Hidden Layer 2 — 50 neurons, ReLU activation
    model.add(Dense(50, activation='relu'))
    
    # 🎯 Output layer — 1 neuron, linear (no activation) for continuous regression
    model.add(Dense(1))
    
    # ⚙️ Compile: Adam optimiser + MSE loss
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

print("✅ regression_model() defined — 2 hidden layers, 50 neurons each")

# %% [markdown]
# ## Part 4 — 📈 Training & Evaluation
#
# ### 🏗️ Instantiating the Model
#
# I call `regression_model()` to create a fresh, randomly initialised instance of the network. Keras uses Glorot Uniform initialisation by default for Dense layers, which scales initial weights according to layer dimensions.
#

# %%
# 🏗️ Instantiate the regression model
model = regression_model()

# 📊 Print architecture summary
model.summary()

# %% [markdown]
# ### 📈 Training with Validation Split
#
# I train for **100 epochs** using 70% of the data for training and reserving **30% for validation**. The `validation_split=0.3` parameter instructs Keras to hold out the last 30% of rows (in order) as a validation set — no shuffling occurs by default, so this is a chronological split.
#
# **Key training hyperparameters:**
#
# | Hyperparameter | Value | Rationale |
# |----------------|-------|-----------|
# | `epochs` | 100 | Enough passes to converge on this small dataset |
# | `validation_split` | 0.3 | 30% held-out data to monitor overfitting |
# | `verbose` | 2 | One line per epoch — readable training log |
#
# The training loop minimises $\mathcal{L}_{\text{MSE}}$ via **backpropagation + Adam**:
#
# $$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}_{\text{MSE}}$$
#
# I watch both `loss` (training MSE) and `val_loss` (validation MSE) to detect overfitting. If `val_loss` starts rising while `loss` keeps falling, the model is memorising training data.
#

# %%
# 📈 Train the model — 100 epochs, 30% validation split
history = model.fit(
    predictors_norm,
    target,
    validation_split=0.3,
    epochs=100,
    verbose=2
)

# %% [markdown]
# ### 📊 Training Curve Analysis
#
# I visualise the training and validation loss curves to assess convergence behaviour and diagnose overfitting or underfitting.
#

# %%
import matplotlib.pyplot as plt

# 📊 Plot training vs validation MSE loss
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(history.history['loss'],     label='📈 Training MSE',   linewidth=2)
ax.plot(history.history['val_loss'], label='🧪 Validation MSE', linewidth=2, linestyle='--')

ax.set_title('🧠 Regression Model — Training vs Validation MSE (2 Hidden Layers)', fontsize=14)
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Squared Error (MPa²)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

final_train_mse = history.history['loss'][-1]
final_val_mse   = history.history['val_loss'][-1]
print(f"\n🎯 Final Training MSE   : {final_train_mse:.2f}")
print(f"🎯 Final Validation MSE : {final_val_mse:.2f}")
print(f"🎯 Final Validation RMSE: {final_val_mse**0.5:.2f} MPa")


# %% [markdown]
# ## Part 5 — 🧪 Extended Architecture: 5 Hidden Layers
#
# ### 🧠 Deeper Network Design
#
# I now extend the architecture to **5 hidden layers**, each with 50 neurons and ReLU activation. Deeper networks have greater representational capacity — they can learn more complex, hierarchical feature interactions in the data.
#
# **Extended architecture:**
#
# ```
# 📥 Input  →  [8 features]
#               ↓
# 🧠 Dense(50, ReLU)   ← Hidden Layer 1
#               ↓
# 🧠 Dense(50, ReLU)   ← Hidden Layer 2
#               ↓
# 🧠 Dense(50, ReLU)   ← Hidden Layer 3
#               ↓
# 🧠 Dense(50, ReLU)   ← Hidden Layer 4
#               ↓
# 🧠 Dense(50, ReLU)   ← Hidden Layer 5
#               ↓
# 🎯 Dense(1, linear)  ← Output: predicted strength (MPa)
# ```
#
# **Why might this perform better?**
# - More layers → higher model capacity → can fit more complex non-linear patterns
# - Potential risk: overfitting on this small (~1,000 row) dataset
#
# **Why train with `validation_split=0.1`?**
# - Using only 10% for validation gives 90% of data to training
# - More training data → better weight estimates → potentially lower bias
#

# %%
def regression_model_deep():
    """
    🏗️ Build and compile a 5-hidden-layer regression neural network.
    
    Architecture:
        Input(n_cols) → Dense(50, ReLU) x5 → Dense(1, linear)
    
    Returns:
        model: Compiled Keras Sequential model
    """
    input_colm = predictors_norm.shape[1]  # Number of input features
    
    model = Sequential()
    
    # 📥 Input layer
    model.add(Input(shape=(input_colm,)))
    
    # 🧠 Hidden Layers 1-5 — each 50 neurons with ReLU
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    
    # 🎯 Output layer — linear activation for regression
    model.add(Dense(1))
    
    # ⚙️ Compile: Adam + MSE
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

print("✅ regression_model_deep() defined — 5 hidden layers, 50 neurons each")

# %% [markdown]
# ### 📈 Training the Deeper Model with 10% Validation Split
#
# I train the 5-layer model for 100 epochs, reserving only 10% of data for validation. This gives the model access to 900 samples during training vs 700 in the baseline.
#

# %%
# 🏗️ Instantiate and inspect the deeper model
model_deep = regression_model_deep()
model_deep.summary()

# %%
# 📈 Train — 100 epochs, 10% validation split
history_deep = model_deep.fit(
    predictors_norm,
    target,
    validation_split=0.1,
    epochs=100,
    verbose=2
)

# %%
# 📊 Compare training curves: shallow vs deep
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: 2-layer model
axes[0].plot(history.history['loss'],     label='📈 Training MSE',   linewidth=2)
axes[0].plot(history.history['val_loss'], label='🧪 Validation MSE', linewidth=2, linestyle='--')
axes[0].set_title('🧠 Baseline — 2 Hidden Layers (val_split=0.3)', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE (MPa²)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: 5-layer model
axes[1].plot(history_deep.history['loss'],     label='📈 Training MSE',   linewidth=2, color='darkorange')
axes[1].plot(history_deep.history['val_loss'], label='🧪 Validation MSE', linewidth=2, linestyle='--', color='red')
axes[1].set_title('🧠 Deep — 5 Hidden Layers (val_split=0.1)', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE (MPa²)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('📊 Training Curve Comparison: Baseline vs Deep Network', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

final_deep_train_mse = history_deep.history['loss'][-1]
final_deep_val_mse   = history_deep.history['val_loss'][-1]
print(f"\n🎯 Deep Model — Final Training MSE   : {final_deep_train_mse:.2f}")
print(f"🎯 Deep Model — Final Validation MSE : {final_deep_val_mse:.2f}")
print(f"🎯 Deep Model — Final Validation RMSE: {final_deep_val_mse**0.5:.2f} MPa")


# %% [markdown]
# ## 📊 Summary
#
# ### 🔑 Key Concepts Covered
#
# | Concept | Detail |
# |---------|--------|
# | 🔢 Dataset | UCI Concrete Compressive Strength — 1030 samples, 8 features, 1 continuous target |
# | 🔢 Normalisation | Z-score standardisation: $x^{\text{norm}} = (x - \mu)/\sigma$ |
# | 🧠 Architecture | Sequential feedforward network — Dense layers stacked linearly |
# | 📈 Activation | ReLU: $\max(0, z)$ — prevents vanishing gradients, fast to compute |
# | 🎯 Output | Linear activation (no function) — allows any real-valued prediction |
# | ⚙️ Optimiser | Adam — adaptive learning rates using first/second moment estimates |
# | ⚙️ Loss | Mean Squared Error (MSE) — penalises large errors quadratically |
# | 🔄 Training | `model.fit()` with `validation_split` — automatic train/val split |
#
# ---
#
# ### 📊 Model Architecture Comparison
#
# | Property | 🏗️ Baseline Model | 🧠 Deep Model |
# |----------|-------------------|---------------|
# | Hidden Layers | 2 | 5 |
# | Neurons per Layer | 50 | 50 |
# | Total Dense Layers | 3 (inc. output) | 6 (inc. output) |
# | Activation (hidden) | ReLU | ReLU |
# | Activation (output) | Linear | Linear |
# | Optimiser | Adam | Adam |
# | Loss | MSE | MSE |
# | Validation Split | 30% | 10% |
# | Training Data Fraction | 70% | 90% |
#
# ---
#
# ### 🔑 Key Observations
#
# - **More hidden layers** increase the model's capacity to learn complex, hierarchical feature relationships — allowing it to fit training data more effectively
# - **Larger training set** (reducing validation split from 30% → 10%) gives the model more examples to learn from, which generally improves generalisation on small datasets
# - On very small datasets (~1,000 rows), deeper networks carry a higher **overfitting risk** — monitored by watching the gap between training and validation MSE curves
# - Keras's **Sequential API** makes it trivial to experiment with depth by simply adding more `Dense` layer calls
#

# %% [markdown]
# ## 🧪 Sandbox — Free Experimentation
#
# This section is my open experimentation space. I use it to explore ideas not covered in the main walkthrough — alternative architectures, different optimisers, learning rate schedules, regularisation, etc.
#

# %%
# 🧪 Experiment 1: Vary neuron count — try 100 neurons per layer
def regression_model_wide():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model_wide = regression_model_wide()
model_wide.summary()

# %%
# 🧪 Experiment 2: Use a different optimiser — SGD with momentum
from keras.optimizers import SGD

def regression_model_sgd():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    # SGD with momentum — classic optimiser for comparison
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mean_squared_error')
    return model

model_sgd = regression_model_sgd()
print("✅ SGD model defined. Train with: model_sgd.fit(predictors_norm, target, epochs=100, validation_split=0.2)")

# %%
# 🧪 Experiment 3: Add Dropout regularisation to combat overfitting
from keras.layers import Dropout

def regression_model_dropout(dropout_rate=0.2):
    """
    🏗️ Regression model with Dropout regularisation.
    
    Dropout randomly zeros out a fraction of neurons during training,
    preventing co-adaptation and reducing overfitting on small datasets.
    """
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rate))   # 🔄 Drop 20% of neurons randomly
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model_dropout = regression_model_dropout(dropout_rate=0.2)
model_dropout.summary()
# ▶️ Train: history_dropout = model_dropout.fit(predictors_norm, target, epochs=100, validation_split=0.3, verbose=2)

# %%
# 🧪 Experiment 4: Your own architecture — edit freely!

# 🔧 Try: different number of layers, neurons, activations, optimisers
def my_custom_model():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    
    # TODO: Add your layers here
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

my_model = my_custom_model()
my_model.summary()
# ▶️ Train: history_custom = my_model.fit(predictors_norm, target, epochs=100, validation_split=0.2, verbose=2)

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
# # 🧠 Activation Functions & Vanishing Gradients
#
# > **IBM AI Engineering Professional Certificate** · Course 2 — Deep Learning & Keras
#
# ---
#
# ## 📋 Overview
#
# In this notebook, I explore two fundamental activation functions used in deep neural networks — **Sigmoid** and **ReLU** — and analyse their derivatives to understand the **vanishing gradient problem**. I also implement the **Tanh** activation function as an extension exercise.
#
# | # | Topic |
# |---|-------|
# | 1 | 🧩 Theory — Sigmoid, ReLU & Vanishing Gradients |
# | 2 | ⚙️ Define Sigmoid & its derivative |
# | 3 | ⚙️ Define ReLU & its derivative |
# | 4 | 📈 Compute & visualise activations and gradients |
# | 5 | 🧪 Practice — Tanh function & comparison with ReLU |
# | 6 | 📊 Summary |
# | 7 | 🔬 Sandbox |
#
# ---
#
# ## 🧩 Theory
#
# ### 📈 Sigmoid Activation Function
#
# The sigmoid function maps any input to a value between **0 and 1**, producing an S-shaped curve:
#
# $$\sigma(z) = \dfrac{1}{1 + e^{-z}}$$
#
# Its derivative (used in backpropagation) is elegantly expressed as:
#
# $$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
#
# | Property | Detail |
# |----------|--------|
# | Output range | (0, 1) |
# | Shape | S-curve |
# | Use case | Binary classification, output probabilities |
# | ⚠️ Limitation | **Vanishing gradient** — for large \|z\|, gradient → 0, slowing learning |
#
# ---
#
# ### ⚡ ReLU Activation Function
#
# ReLU (Rectified Linear Unit) is the most widely used activation in hidden layers:
#
# $$f(x) = \max(0, x)$$
#
# Its derivative is:
#
# $$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$
#
# | Property | Detail |
# |----------|--------|
# | Output range | [0, ∞) |
# | Shape | Linear for x > 0, zero otherwise |
# | Use case | Hidden layers in deep networks |
# | ✅ Advantage | **No vanishing gradient** for positive inputs |
# | ⚠️ Limitation | **Dead neurons** — neurons stuck at 0 for all negative inputs |
#
# ---
#
# ## ⚙️ Part 1 — Setup & Imports
#

# %%
# 📦 Install required libraries (run once if needed)
# # !pip install numpy==2.0.2
# # !pip install matplotlib==3.9.2


# %%
import numpy as np
import matplotlib.pyplot as plt

print("✅ Libraries imported successfully!")
print(f"📦 NumPy version:      {np.__version__}")
import matplotlib
print(f"📦 Matplotlib version: {matplotlib.__version__}")


# %% [markdown]
# ---
#
# ## ⚙️ Part 2 — Sigmoid Activation Function
#
# ### 📈 Sigmoid Function
#
# $$\sigma(z) = \dfrac{1}{1 + e^{-z}}$$
#

# %%
def sigmoid(z):
    """📈 Sigmoid activation: maps any input to (0, 1)."""
    return 1 / (1 + np.exp(-z))

# 🧪 Quick test
test_vals = np.array([-5, -1, 0, 1, 5])
print("📥 Input values:    ", test_vals)
print("📤 Sigmoid outputs: ", np.around(sigmoid(test_vals), 4))


# %% [markdown]
# ### 📉 Sigmoid Derivative
#
# The gradient of sigmoid — used during backpropagation:
#
# $$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$$
#
# > ⚠️ Notice how this approaches **zero** for large positive or negative inputs — this is the **vanishing gradient problem**.
#

# %%
def sigmoid_derivative(z):
    """📉 Sigmoid gradient: σ(z) · (1 - σ(z))."""
    return sigmoid(z) * (1 - sigmoid(z))

print("📥 Input values:             ", test_vals)
print("📤 Sigmoid gradient outputs: ", np.around(sigmoid_derivative(test_vals), 4))


# %% [markdown]
# ---
#
# ## ⚙️ Part 3 — ReLU Activation Function
#
# ### ⚡ ReLU Function
#
# $$f(x) = \max(0, x)$$
#

# %%
def relu(z):
    """⚡ ReLU activation: max(0, z)."""
    return np.maximum(0, z)

print("📥 Input values:  ", test_vals)
print("📤 ReLU outputs:  ", relu(test_vals))


# %% [markdown]
# ### ⚡ ReLU Derivative
#
# The gradient of ReLU is binary — either **1** (positive inputs) or **0** (negative inputs):
#
# $$f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$
#
# > ✅ Unlike sigmoid, the gradient stays at **1** for all positive inputs — no vanishing gradient!
#

# %%
def relu_derivative(z):
    """⚡ ReLU gradient: 1 where z > 0, else 0."""
    return np.where(z > 0, 1, 0)

print("📥 Input values:           ", test_vals)
print("📤 ReLU gradient outputs:  ", relu_derivative(test_vals))


# %% [markdown]
# ---
#
# ## 📈 Part 4 — Compute & Visualise Activations and Gradients
#
# I generate **400 input values** between -10 and 10, compute both activations and their gradients, then visualise them side by side.
#

# %%
# 🔢 Generate 400 synthetic input values between -10 and 10
z = np.linspace(-10, 10, 400)

# 📊 Compute gradients
sigmoid_grad = sigmoid_derivative(z)
relu_grad    = relu_derivative(z)

print(f"✅ Generated {len(z)} input values from {z[0]:.1f} to {z[-1]:.1f}")
print(f"📈 Sigmoid gradient range: [{sigmoid_grad.min():.4f}, {sigmoid_grad.max():.4f}]")
print(f"⚡ ReLU gradient range:    [{relu_grad.min():.4f}, {relu_grad.max():.4f}]")


# %%
# 📊 Visualise activation functions and their gradients
plt.figure(figsize=(14, 6))

# ── Sigmoid ──────────────────────────────────────────
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z),  label='Sigmoid Activation', color='steelblue', linewidth=2)
plt.plot(z, sigmoid_grad, label='Sigmoid Derivative', color='tomato', linestyle='--', linewidth=2)
plt.title('📈 Sigmoid Activation & Gradient', fontsize=13, fontweight='bold')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

# ── ReLU ─────────────────────────────────────────────
plt.subplot(1, 2, 2)
plt.plot(z, relu(z),  label='ReLU Activation', color='seagreen', linewidth=2)
plt.plot(z, relu_grad, label='ReLU Derivative', color='tomato', linestyle='--', linewidth=2)
plt.title('⚡ ReLU Activation & Gradient', fontsize=13, fontweight='bold')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

plt.suptitle('🧠 Activation Functions & Vanishing Gradients', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n🔍 Key Observation:")
print("  📈 Sigmoid gradient approaches 0 at extremes → vanishing gradient problem")
print("  ⚡ ReLU gradient stays at 1 for positive inputs → no vanishing gradient")


# %% [markdown]
# ---
#
# ## 🧪 Part 5 — Practice: Tanh Function & Comparison with ReLU
#
# ### 🧩 Theory — Tanh Activation
#
# The hyperbolic tangent function is similar to sigmoid but maps inputs to **(-1, 1)**:
#
# $$\tanh(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$$
#
# Its derivative:
#
# $$\tanh'(z) = 1 - \tanh^2(z)$$
#
# > 💡 Tanh is **zero-centred** (unlike sigmoid), which often leads to faster convergence during training.
#
# ### ⚙️ Practice Exercise 1 — Implement Tanh
#

# %%
def tanh(z):
    """🔄 Tanh activation: maps input to (-1, 1)."""
    return np.tanh(z)

def tanh_derivative(z):
    """🔄 Tanh gradient: 1 - tanh²(z)."""
    return 1 - np.tanh(z) ** 2

# 🧪 Quick test
print("📥 Input values:           ", test_vals)
print("📤 Tanh outputs:           ", np.around(tanh(test_vals), 4))
print("📤 Tanh gradient outputs:  ", np.around(tanh_derivative(test_vals), 4))


# %% [markdown]
# ### 📊 Practice Exercise 2 — Plot Tanh vs ReLU
#
# I now compare ReLU and Tanh activations and their gradients over 100 values between -5 and 5.
#

# %%
# 🔢 Generate 100 synthetic input values between -5 and 5
z = np.linspace(-5, 5, 100)

# 📊 Compute gradients
tanh_grad = tanh_derivative(z)
relu_grad  = relu_derivative(z)

# 📊 Visualise ReLU vs Tanh
plt.figure(figsize=(14, 6))

# ── ReLU ─────────────────────────────────────────────
plt.subplot(1, 2, 1)
plt.plot(z, relu(z),  label='ReLU Activation', color='seagreen', linewidth=2)
plt.plot(z, relu_grad, label='ReLU Derivative', color='tomato', linestyle='--', linewidth=2)
plt.title('⚡ ReLU Activation & Gradient', fontsize=13, fontweight='bold')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

# ── Tanh ─────────────────────────────────────────────
plt.subplot(1, 2, 2)
plt.plot(z, tanh(z),  label='Tanh Activation', color='mediumpurple', linewidth=2)
plt.plot(z, tanh_grad, label='Tanh Derivative', color='tomato', linestyle='--', linewidth=2)
plt.title('🔄 Tanh Activation & Gradient', fontsize=13, fontweight='bold')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

plt.suptitle('⚡ ReLU vs 🔄 Tanh — Activation & Gradient Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n🔍 Key Observations:")
print("  ⚡ ReLU: gradient is 1 for positive, 0 for negative (dead neuron risk)")
print("  🔄 Tanh: gradient approaches 0 at extremes (mild vanishing gradient)")
print("  🎯 Tanh is zero-centred → often converges faster than sigmoid")


# %% [markdown]
# ---
#
# ## 📊 Summary
#
# | Activation | Formula | Output Range | Gradient | Vanishing Gradient | Best Used For |
# |-----------|---------|-------------|----------|-------------------|--------------|
# | 📈 Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | (0, 1) | $\sigma(z)(1-\sigma(z))$ | ⚠️ Severe | Output layer (binary) |
# | ⚡ ReLU | $\max(0, z)$ | [0, ∞) | 1 if z>0, else 0 | ✅ None | Hidden layers |
# | 🔄 Tanh | $\tanh(z)$ | (-1, 1) | $1 - \tanh^2(z)$ | ⚠️ Mild | Hidden layers (RNNs) |
#
# ### 🎯 Key Takeaways
#
# - **Vanishing gradient** occurs when gradients become near-zero, preventing weights from updating — a major problem in deep networks
# - **Sigmoid** suffers most — gradients collapse at both extremes
# - **ReLU** solves this for positive inputs but can create **dead neurons**
# - **Tanh** is zero-centred and often preferred over sigmoid in hidden layers
# - In practice, **ReLU** and its variants (Leaky ReLU, ELU) dominate modern deep learning architectures
#
# ---
#
# ## 🔬 Sandbox — Free Experimentation
#

# %%
# 🧪 Experiment 1 — Leaky ReLU (fixes dead neuron problem)
def leaky_relu(z, alpha=0.01):
    """⚡ Leaky ReLU: allows small gradient for negative inputs."""
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

z = np.linspace(-5, 5, 100)

plt.figure(figsize=(10, 4))
plt.plot(z, relu(z),         label='ReLU',       color='seagreen',   linewidth=2)
plt.plot(z, leaky_relu(z),   label='Leaky ReLU', color='darkorange',  linewidth=2, linestyle='--')
plt.plot(z, tanh(z),         label='Tanh',        color='mediumpurple',linewidth=2, linestyle=':')
plt.title('🔬 Activation Function Comparison', fontsize=13, fontweight='bold')
plt.xlabel('Input Value (z)')
plt.ylabel('Output')
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()


# %%
# 🧪 Experiment 2 — Compare all gradients side by side
z = np.linspace(-5, 5, 100)

plt.figure(figsize=(10, 4))
plt.plot(z, sigmoid_derivative(z),      label='Sigmoid gradient',      color='steelblue',   linewidth=2)
plt.plot(z, relu_derivative(z),         label='ReLU gradient',         color='seagreen',    linewidth=2)
plt.plot(z, tanh_derivative(z),         label='Tanh gradient',         color='mediumpurple',linewidth=2)
plt.plot(z, leaky_relu_derivative(z),   label='Leaky ReLU gradient',   color='darkorange',  linewidth=2, linestyle='--')
plt.title('📉 Gradient Comparison — Vanishing Gradient Visualised', fontsize=13, fontweight='bold')
plt.xlabel('Input Value (z)')
plt.ylabel('Gradient Value')
plt.legend()
plt.grid(alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

print("\n🎯 Observation: Sigmoid gradient is near-zero for |z| > 3 → vanishing gradient")
print("⚡ ReLU and Leaky ReLU maintain non-zero gradients → better for deep networks")


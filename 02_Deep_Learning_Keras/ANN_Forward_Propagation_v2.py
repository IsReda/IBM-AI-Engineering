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
# # 🧠 Artificial Neural Networks — Forward Propagation
#
# > **IBM AI Engineering Professional Certificate** · Deep Learning Fundamentals
#
# ---
#
# ## 📋 Overview
#
# In this notebook, I build a neural network **from scratch** and implement **forward propagation** to understand exactly how predictions flow through a network. While deep learning libraries like TensorFlow and PyTorch handle this automatically in production, building it manually sharpens my intuition for what happens at each layer.
#
# ---
#
# ## 🎯 What I Cover
#
# | # | Topic |
# |---|-------|
# | 1 | 🔢 Manual forward propagation on a simple 2-input network |
# | 2 | 🏗️ Generalised network initialisation (any depth & width) |
# | 3 | ➕ Weighted sum computation |
# | 4 | 📈 Sigmoid activation function |
# | 5 | 🔄 Full end-to-end forward propagation |
# | 6 | 🧪 Sandbox experiments with custom architectures |
#
# ---
#
# ## 🧩 Theory: How Forward Propagation Works
#
# At every node in the network, two operations happen sequentially:
#
# **Step 1 — Weighted Sum:**
# $$z = \sum_{i} (x_i \cdot w_i) + b$$
#
# **Step 2 — Activation (Sigmoid):**
# $$a = \sigma(z) = \dfrac{1}{1 + e^{-z}}$$
#
# The activations of one layer become the inputs to the next — this cascade is **forward propagation**.
#
# The simple network I start with: **2 inputs → 1 hidden layer (2 nodes) → 1 output node**
#
# ![Simple Neural Network](http://cocl.us/neural_network_example)
#

# %% [markdown]
# ---
#
# ## 🔢 Part 1 — Manual Forward Propagation
#
# I wire up a small network by hand to build intuition before generalising.
#
# ### ⚙️ Step 1 · Initialise Weights & Biases

# %%
import numpy as np

# 🔧 6 weights (2 hidden nodes × 2 inputs + 1 output node × 2 inputs)
# 🔧 3 biases  (one per node)
weights = np.around(np.random.uniform(size=6), decimals=2)
biases  = np.around(np.random.uniform(size=3), decimals=2)

print("⚖️  Weights:", weights)
print("📌 Biases: ", biases)


# %% [markdown]
# ### 📥 Step 2 · Define Inputs

# %%
x_1 = 0.5   # first input feature
x_2 = 0.85  # second input feature

print(f"📥 x₁ = {x_1},  x₂ = {x_2}")


# %% [markdown]
# ### ➕ Step 3 · Weighted Sum — Hidden Layer Node 1
#
# $$z_{1,1} = x_1 \cdot w_0 + x_2 \cdot w_1 + b_0$$

# %%
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print(f"➕ Weighted sum at hidden node 1:  {z_11:.4f}")


# %% [markdown]
# ### ➕ Step 4 · Weighted Sum — Hidden Layer Node 2
#
# $$z_{1,2} = x_1 \cdot w_2 + x_2 \cdot w_3 + b_1$$

# %%
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print(f"➕ Weighted sum at hidden node 2:  {np.around(z_12, decimals=4)}")


# %% [markdown]
# ### 📈 Step 5 · Sigmoid Activation — Hidden Layer
#
# $$a = \sigma(z) = \dfrac{1}{1 + e^{-z}}$$
#
# The sigmoid squashes any value into the range **(0, 1)** — perfect for probabilities.

# %%
a_11 = 1.0 / (1.0 + np.exp(-z_11))
a_12 = 1.0 / (1.0 + np.exp(-z_12))

print(f"📈 Activation at hidden node 1:  {np.around(a_11, decimals=4)}")
print(f"📈 Activation at hidden node 2:  {np.around(a_12, decimals=4)}")


# %% [markdown]
# ### 📤 Step 6 · Output Node — Weighted Sum & Activation

# %%
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
a_2 = 1.0 / (1.0 + np.exp(-z_2))

print(f"➕ Weighted sum at output node:   {np.around(z_2, decimals=4)}")
print(f"🎯 Network prediction (output):   {np.around(a_2, decimals=4)}")


# %% [markdown]
# ---
#
# ## 🏗️ Part 2 — Generalised Network Initialisation
#
# Computing this manually doesn't scale beyond toy examples. I now build a **generalised network** that handles any architecture.
#
# ![General Neural Network](http://cocl.us/general_neural_network)
#
# > **Architecture I use:** any number of inputs → any hidden layer depth & width → any output size
#

# %%
import numpy as np

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    """
    🏗️ Initialise a fully-connected neural network with random weights and biases.

    Args:
        num_inputs        (int)  : number of input features
        num_hidden_layers (int)  : number of hidden layers
        num_nodes_hidden  (list) : node count for each hidden layer
        num_nodes_output  (int)  : number of output nodes

    Returns:
        network (dict): nested dict → layer → node → {weights, bias}
    """
    num_nodes_previous = num_inputs
    network = {}

    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes  = num_nodes_output
        else:
            layer_name = f'layer_{layer + 1}'
            num_nodes  = num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = f'node_{node + 1}'
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias'   : np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network



# %%
# 🧪 Test: 5 inputs → 3 hidden layers [3, 2, 3] nodes → 1 output
small_network = initialize_network(5, 3, [3, 2, 3], 1)

print("✅ Network initialised successfully!")
print("📐 Layers:", list(small_network.keys()))


# %% [markdown]
# ---
#
# ## ➕ Part 3 — Weighted Sum Function
#
# I encapsulate the weighted sum formula as a clean, reusable function.

# %%
def compute_weighted_sum(inputs, weights, bias):
    """➕ Compute the weighted sum: dot(inputs, weights) + bias."""
    return np.sum(inputs * weights) + bias



# %%
# 📥 Generate 5 random inputs for small_network
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
print(f"📥 Inputs to the network: {inputs}")

# ➕ Weighted sum at first node of first hidden layer
node_weights = small_network['layer_1']['node_1']['weights']
node_bias    = small_network['layer_1']['node_1']['bias']

weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print(f"➕ Weighted sum at layer_1 / node_1: {np.around(weighted_sum[0], decimals=4)}")


# %% [markdown]
# ---
#
# ## 📈 Part 4 — Sigmoid Activation Function
#
# The sigmoid function introduces **non-linearity**, enabling the network to learn complex patterns.

# %%
def node_activation(weighted_sum):
    """📈 Apply sigmoid activation: σ(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

# 🧪 Activation at first node of first hidden layer
node_output = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
print(f"📈 Activation at layer_1 / node_1: {np.around(node_output[0], decimals=4)}")


# %% [markdown]
# ---
#
# ## 🔄 Part 5 — Full Forward Propagation
#
# I now assemble the complete pipeline. `forward_propagate` passes data through every layer, left to right, from inputs to predictions.
#
# **Algorithm:**
# 1. Start with raw inputs as the first layer's input
# 2. At each layer: compute weighted sums → apply activation
# 3. Pass that layer's outputs as the next layer's inputs
# 4. Repeat until the output layer is reached
#

# %%
def forward_propagate(network, inputs):
    """
    🔄 Propagate inputs through all layers; return output layer predictions.

    Args:
        network (dict) : initialised network from initialize_network()
        inputs  (list) : input feature values

    Returns:
        predictions (list): activation values at the output layer
    """
    layer_inputs = list(inputs)

    for layer in network:
        layer_data    = network[layer]
        layer_outputs = []

        for layer_node in layer_data:
            node_data   = layer_data[layer_node]
            node_output = node_activation(
                compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias'])
            )
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print(f"  🔁 {layer} outputs: {layer_outputs}")

        layer_inputs = layer_outputs

    return layer_outputs



# %%
# 🎯 Run forward propagation on small_network
print("🔄 Forward propagation through small_network:\n")
predictions = forward_propagate(small_network, inputs)
print(f"\n🎯 Final prediction: {np.around(predictions[0], decimals=4)}")


# %% [markdown]
# ---
#
# ## 🧪 Part 6 — End-to-End Test
#
# I build and run a fresh network from scratch — confirming the full pipeline works correctly.

# %%
# 🏗️ Build: 5 inputs → [2, 3, 2] hidden nodes → 3 outputs
my_network = initialize_network(5, 3, [2, 3, 2], 3)

# 📥 Fresh random inputs
inputs = np.around(np.random.uniform(size=5), decimals=2)
print(f"📥 Inputs: {inputs}\n")

# 🔄 Forward propagate
print("🔄 Propagating through layers:\n")
predictions = forward_propagate(my_network, inputs)
print(f"\n🎯 Predicted values: {predictions}")


# %% [markdown]
# ---
#
# ## 📊 Summary
#
# | Component | Function | Role |
# |-----------|----------|------|
# | `initialize_network()` | 🏗️ Build architecture | Creates weights & biases for any topology |
# | `compute_weighted_sum()` | ➕ Linear transform | $z = \sum x_i w_i + b$ |
# | `node_activation()` | 📈 Non-linearity | $\sigma(z) = 1/(1+e^{-z})$ |
# | `forward_propagate()` | 🔄 End-to-end inference | Chains all layers input → output |
#
# **Key insight:** forward propagation is a **repeated sequence** of dot products + bias additions + sigmoid activations, cascaded from the input layer through to the output. Deep learning libraries (TensorFlow, PyTorch) optimise this exact process with GPU acceleration and automatic differentiation.
#
# ---
#
# ## 🧪 Sandbox — Free Experimentation
#

# %%
# 🧪 Custom network 1 — try your own architecture
custom_net_1   = initialize_network(3, 2, [4, 4], 2)
custom_inputs_1 = np.around(np.random.uniform(size=3), decimals=2)

print("🔬 Custom Network 1")
print(f"📥 Inputs: {custom_inputs_1}\n")
preds = forward_propagate(custom_net_1, custom_inputs_1)
print(f"\n🎯 Predictions: {preds}")


# %%
# 🧪 Custom network 2 — deeper architecture
custom_net_2    = initialize_network(6, 4, [5, 4, 3, 2], 1)
custom_inputs_2 = np.around(np.random.uniform(size=6), decimals=2)

print("🔬 Custom Network 2")
print(f"📥 Inputs: {custom_inputs_2}\n")
preds = forward_propagate(custom_net_2, custom_inputs_2)
print(f"\n🎯 Predictions: {preds}")


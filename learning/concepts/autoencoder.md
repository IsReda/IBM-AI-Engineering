---
type: concept
topic: autoencoder
module: "[[03_Deep_Learning_Keras_TensorFlow]]"
status: 🌱 Seedling
tags: [ml, keras, tensorflow, deep-learning, unsupervised, reconstruction, course/03]
---

# Autoencoder

## In my own words

An autoencoder is a neural network trained to reconstruct its own input — the target equals the input ($y = x$). It's split into two halves:

- **Encoder**: compresses the input into a smaller **bottleneck** representation (latent space)
- **Decoder**: reconstructs the original input from the bottleneck

Because the network must pass everything through a narrow bottleneck, it's forced to learn a compressed, structured representation of the data — capturing only the most essential features. No labels required: the loss signal is purely the reconstruction error.

## Simple analogy

**Telecom / RF 📡:** An autoencoder is the neural equivalent of a codec (coder-decoder):
- **Encoder** ≈ source coder — compress a signal into the fewest bits that still capture its key structure (analogous to speech codecs like CELP, or image codecs like JPEG)
- **Bottleneck** ≈ the compressed bitstream — a compact, information-dense representation
- **Decoder** ≈ source decoder — reconstruct the signal from the compressed representation

An autoencoder trained to minimise MSE reconstruction loss is essentially learning the optimal codec for the data it sees. High reconstruction error for an anomalous input is equivalent to a codec failing to compress an unexpected signal — because the codec was tuned for normal signals, not anomalies.

## Architecture

```python
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model

# Simple convolutional autoencoder (28×28 grayscale images)
input_layer = Input(shape=(28, 28, 1))

# Encoder: learn compact representation
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

# Decoder: reconstruct from representation
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(encoded)

# Model: input → reconstruction
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile: target = input (reconstruction task)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train: X_train is both input and target
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32)
```

### Strided autoencoder (true bottleneck)

```python
input_layer = Input(shape=(28, 28, 1))

# Encoder: 28×28 → 14×14 → 7×7 (downsampling via strides)
x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(input_layer)
encoded = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x)
# bottleneck: 7×7×16 = 784 → 784 values (same as input but structured)

# Decoder: 7×7 → 14×14 → 28×28 (upsampling via transpose conv)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2)(encoded)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=2)(x)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```

## Loss function

**MSE (Mean Squared Error)** is standard for continuous-valued reconstruction:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2$$

Where $x_i$ is the original image and $\hat{x}_i$ is the reconstruction.

- Use `activation='sigmoid'` in the decoder output → data must be in $[0, 1]$ range
- Use `activation='tanh'` in the decoder output → data must be in $[-1, 1]$ range (rescale with `x = x * 2 - 1`)

## Applications

| Application | How autoencoders help |
|---|---|
| **Dimensionality reduction** | Bottleneck = compact feature vector, like PCA but non-linear |
| **Denoising** | Train with noisy input → clean target; learns to filter noise |
| **Anomaly detection** | Normal data reconstructs well; anomalies have high MSE |
| **Pretraining** | Learn features unsupervised, then attach classifier head |
| **Image compression** | Encoder compresses, decoder reconstructs (research use) |
| **Data generation** | Variational autoencoders ([[vae]]) can sample from latent space |

### Anomaly detection pattern

```python
# Train on normal data only
autoencoder.fit(X_normal, X_normal, epochs=50)

# Reconstruct test data
X_reconstructed = autoencoder.predict(X_test)

# Compute per-sample reconstruction error
errors = np.mean((X_test - X_reconstructed) ** 2, axis=(1, 2, 3))

# Flag high-error samples as anomalies
threshold = np.percentile(errors, 95)  # top 5% are anomalies
anomalies = X_test[errors > threshold]
```

Logic: the autoencoder learns the manifold of normal data. An anomalous input doesn't lie on this manifold, so the decoder can't reconstruct it well → high MSE → flagged.

## Key concepts

| Concept | Detail |
|---|---|
| **Bottleneck** | Smallest layer in the network; forces compression |
| **Latent space** | The representation at the bottleneck |
| **Reconstruction loss** | How well $\hat{x}$ matches $x$ — the only training signal |
| **Target = Input** | No labels — purely unsupervised |
| **Encoder** | Input → latent code (compression) |
| **Decoder** | Latent code → reconstruction (decompression) |
| **Undercomplete** | Bottleneck smaller than input (standard autoencoder) |
| **Overcomplete** | Bottleneck larger than input — needs regularisation (sparse AE) |

## Autoencoder variants

| Variant | What changes | Why |
|---|---|---|
| **Denoising AE** | Corrupt input, reconstruct clean target | Robust features, noise removal |
| **Sparse AE** | Add L1 penalty on bottleneck activations | Encourages few active units (interpretable) |
| **Variational AE ([[vae]])** | Bottleneck = $\mu$, $\sigma$ of a distribution | Enables sampling new data |
| **Convolutional AE** | Uses Conv2D + Conv2DTranspose | Works on images |
| **LSTM AE** | Uses LSTM layers | Works on sequences |

## Connected concepts

- [[transpose_convolution]] — primary decoder building block for image autoencoders
- [[conv2d]] — primary encoder building block
- [[keras_functional_api]] — `Model(inputs=..., outputs=...)` is the standard wiring pattern
- [[vae]] — probabilistic extension; enables generation
- [[gan]] — alternative generative model; adversarial instead of reconstruction loss
- [[dropout]] — can be added to encoder for regularisation
- [[pca]] — linear autoencoder is equivalent to PCA; non-linear generalises it
- Module: [[03_Deep_Learning_Keras_TensorFlow]]

## Open questions

- [ ] What exactly does the latent space "look like" for a well-trained autoencoder on images?
- [ ] Why is the VAE [[vae]] better for generation than a standard autoencoder?
- [ ] How do you choose the bottleneck size — is there a principled method?
- [ ] For anomaly detection, what's the best threshold strategy?

## My confidence

- [x] 🌱 Just encountered it
- [ ] 🌿 I can explain it roughly
- [ ] 🌳 I can explain it clearly and apply it

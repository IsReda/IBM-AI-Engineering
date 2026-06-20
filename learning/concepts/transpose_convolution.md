---
type: concept
topic: transpose_convolution
module: "[[03_Deep_Learning_Keras_TensorFlow]]"
status: 🌱 Seedling
tags: [ml, keras, tensorflow, deep-learning, cnn, upsampling, course/03]
---

# Transpose Convolution (Conv2DTranspose)

## In my own words

A regular `Conv2D` compresses spatial information — a 28×28 image becomes a 14×14 feature map when stride=2. **Transpose convolution** (`Conv2DTranspose`) does the opposite: it expands a small feature map back to a larger spatial size, using **learned weights** to decide how to fill in the upsampled space.

It's also called *deconvolution* (technically incorrect — it doesn't invert convolution mathematically) or *fractionally-strided convolution* (more precise — with stride > 1, it inserts zeros between input elements before applying the kernel).

## Simple analogy

**Telecom / RF 📡:** A regular convolution is like a bank of correlators that compress a received waveform into a vector of match scores — one score per filter. Transpose convolution is the reverse: like a pulse-shaping filter that reconstructs a continuous waveform from discrete samples. But unlike a fixed sinc filter, the reconstruction weights are *learned from data* — the network decides the optimal interpolation kernel for the task at hand.

In communications terms: Conv2D ≈ analysis filter bank (signal → subband coefficients). Conv2DTranspose ≈ synthesis filter bank (coefficients → reconstructed signal).

## The formula

### Output size

For `Conv2DTranspose` with stride $s$, kernel $k$, `padding='same'`:

$$H_{out} = H_{in} \times s, \quad W_{out} = W_{in} \times s$$

With `stride=1` (default): $H_{out} = H_{in}$ — same spatial size, acts as a feature mixer.  
With `stride=2`: $H_{out} = 2 \times H_{in}$ — true spatial upsampling.

### How it operates (stride=2 example)

Input pixels are spread out by inserting $(s-1)$ zeros between them, then convolved with the kernel:

```
Input:   [a, b, c]       (stride=2 transpose)
Spread:  [a, 0, b, 0, c]
Kernel:  [k1, k2, k3]
Output:  [a·k1, a·k2+b·k1, a·k3+b·k2+c·k1, b·k3+c·k2, c·k3]
```

The kernel weights are learned — the network finds the best interpolation strategy.

## Architecture / How it works

```python
from tensorflow.keras.layers import Conv2DTranspose, Input, Conv2D
from tensorflow.keras.models import Model

# Minimal encoder-decoder
input_layer = Input(shape=(28, 28, 1))

# Encoder: 28×28 → 14×14 (stride=2 downsampling)
encoded = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(input_layer)

# Decoder: 14×14 → 28×28 (stride=2 upsampling via transpose conv)
decoded = Conv2DTranspose(1, (3, 3), strides=2, activation='sigmoid', padding='same')(encoded)

model = Model(inputs=input_layer, outputs=decoded)

# Key parameters
Conv2DTranspose(
    filters=1,               # number of output feature maps
    kernel_size=(3, 3),      # learned reconstruction kernel
    strides=(2, 2),          # upsampling factor: output = 2× input size
    activation='sigmoid',    # match output range to data range
    padding='same'           # maintain alignment (vs 'valid' which can differ)
)
```

## Transpose conv vs alternatives

| Method | Trainable weights | Learns upsampling? | Artefacts | Speed |
|---|---|---|---|---|
| `UpSampling2D(nearest)` | ❌ None | ❌ Fixed copy | Blocky edges | Fastest |
| `UpSampling2D(bilinear)` | ❌ None | ❌ Fixed interp | Smooth but blurry | Fast |
| `Conv2DTranspose` | ✅ Yes | ✅ Learned | Checkerboard* | Medium |
| `UpSampling2D` + `Conv2D` | Partial (conv only) | Partial | Minimal | Fast |

*Checkerboard artefacts occur when `stride > kernel_size` — uneven overlap causes periodic patterns. Fix: use `kernel_size = stride` (e.g. both = 2).

## When to use

| Architecture | Role of Conv2DTranspose |
|---|---|
| **Autoencoder** | Decoder — expand bottleneck back to input size |
| **GAN generator** | Upsample latent vector → full-resolution image |
| **U-Net / segmentation** | Decoder path — upsample skip-connection features |
| **Super-resolution** | Expand low-res input to high-res output |
| **Image reconstruction** | Recover original from compressed representation |

## Key points

| Property | Detail |
|---|---|
| `strides=1` | Feature mixer — no spatial change |
| `strides=2` | 2× spatial upsampling |
| `padding='same'` | Output size = input × stride |
| `padding='valid'` | Output size = input × stride + (kernel - 1) |
| Decoder activation | Must match data range: sigmoid for $[0,1]$, tanh for $[-1,1]$ |
| Checkerboard fix | Set `kernel_size = stride` |

## Connected concepts

- [[autoencoder]] — transpose conv is the decoder building block
- [[conv2d]] — the downsampling counterpart
- [[gan]] — GAN generators use transpose conv to grow from latent vector to image
- [[cnn]] — broader CNN context
- [[keras_functional_api]] — typical usage pattern for building encoder-decoders
- Module: [[03_Deep_Learning_Keras_TensorFlow]]

## Open questions

- [ ] When should I prefer `UpSampling2D + Conv2D` over `Conv2DTranspose`?
- [ ] How does pixel shuffle / sub-pixel convolution compare to transpose conv for super-resolution?
- [ ] Can checkerboard artefacts always be fixed by matching kernel to stride, or do other factors cause them?

## My confidence

- [x] 🌱 Just encountered it
- [ ] 🌿 I can explain it roughly
- [ ] 🌳 I can explain it clearly and apply it

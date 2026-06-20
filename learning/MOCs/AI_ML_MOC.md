---
type: MOC
topic: AI / Machine Learning
updated: 2026-06-06
tags: [MOC, ai-engineering, IBM-certificate]
---

# 🗺️ AI & Machine Learning — Map of Content

> This is your **home page** for all AI/ML learning.
> Every module, every concept, every open question — all linked from here.
> Update it as you progress through the course.

**Course:** IBM AI Engineering Professional Certificate (13 courses · ~166 hours)
**Portfolio:** [[IsReda]] · github.com/IsReda
**You:** [[me]] ← your full context

---

## 📊 Course Progress

| #   | Folder                                            | Coursera Course Name                                        | Hours | Status         | Notes                                                                    |
| --- | ------------------------------------------------- | ----------------------------------------------------------- | ----- | -------------- | ------------------------------------------------------------------------ |
| 01  | [[01_ML_with_Python]]                             | Machine Learning with Python                                | 20h   | ✅ Done         | scikit-learn, regression, classification, clustering, PCA, pipelines     |
| 02  | [[02_Intro_Deep_Learning_Neural_Networks_Keras]]  | Introduction to Deep Learning & Neural Networks with Keras  | 10h   | ✅ Done         | ANN, backprop, activation functions, CNNs, RNNs, transformers intro, Functional API      |
| 03  | [[03_Deep_Learning_Keras_TensorFlow]]             | Deep Learning with Keras and TensorFlow                     | 23h   | 🔄 In Progress | Custom layers, advanced CNNs, GANs, autoencoders, reinforcement learning |
| 04  | [[04_Intro_Neural_Networks_PyTorch]]              | Introduction to Neural Networks and PyTorch                 | 19h   | ⬜ Not started  | Tensors, autograd, DataLoader, linear & logistic regression in PyTorch   |
| 05  | [[05_Deep_Learning_PyTorch]]                      | Deep Learning with PyTorch                                  | 19h   | ⬜ Not started  | Shallow & deep nets, dropout, batch norm, CNNs, transfer learning        |
| 06  | [[06_AI_Capstone_Deep_Learning]]                  | AI Capstone Project with Deep Learning                      | 15h   | ⬜ Not started  | Full DL pipeline, CNNs, vision transformers, geospatial classification   |
| 07  | [[07_GenAI_LLMs_Architecture_Data_Prep]]          | Generative AI and LLMs: Architecture and Data Preparation   | 6h    | ⬜ Not started  | RNNs, transformers, VAEs, GANs, diffusion models, tokenization           |
| 08  | [[08_GenAI_Foundational_Models_NLP]]              | Gen AI Foundational Models for NLP & Language Understanding | 10h   | ⬜ Not started  | Word2Vec, embeddings, N-Grams, encoder-decoder RNNs, seq2seq             |
| 09  | [[09_GenAI_Language_Modeling_Transformers]]       | Generative AI Language Modeling with Transformers           | 9h    | ⬜ Not started  | Attention, positional encoding, GPT vs BERT, Hugging Face                |
| 10  | [[10_GenAI_Engineering_Fine_Tuning_Transformers]] | Generative AI Engineering and Fine-Tuning Transformers      | 8h    | ⬜ Not started  | PEFT, LoRA, QLoRA, Hugging Face fine-tuning                              |
| 11  | [[11_GenAI_Advanced_Fine_Tuning_LLMs]]            | Generative AI Advanced Fine-Tuning for LLMs                 | 9h    | ⬜ Not started  | RLHF, DPO, PPO, instruction tuning, reward modeling                      |
| 12  | [[12_AI_Agents_RAG_LangChain]]                    | Fundamentals of AI Agents Using RAG and LangChain           | 9h    | ⬜ Not started  | RAG, LangChain, prompt engineering, AI agents, vector databases          |
| 13  | [[13_Project_GenAI_Apps_RAG_LangChain]]           | Project: Generative AI Applications with RAG and LangChain  | 9h    | ⬜ Not started  | Vector DB, retriever, Gradio UI, QA bot with LangChain + LLM             |

**Total:** ~166 hours · 4 months at 10h/week

---

## 🧠 Concept Map

### Foundations (Course 01)
- [[supervised_learning]] — labeled data, known output
- [[unsupervised_learning]] — find patterns, no labels
- [[reinforcement_learning]] — reward/penalty, trial & error
- [[overfitting]] — memorises training data, fails on new
- [[underfitting]] — model too simple, misses the pattern
- [[train_test_split]] — why we hold out data
- [[cross_validation]] — K-fold, more reliable evaluation
- [[gridsearchcv]] — hyperparameter tuning
- [[ml_pipeline]] — chaining steps, preventing data leakage

### Regression (Course 01)
- [[linear_regression]] — y = b₀ + b₁x
- [[multiple_linear_regression]] — many features
- [[polynomial_regression]] — curves, x², x³
- [[ridge_lasso_regression]] — regularisation (L1/L2)
- [[evaluation_metrics_regression]] — MAE, MSE, RMSE, R²

### Classification (Course 01)
- [[logistic_regression]] — outputs probability
- [[k_nearest_neighbors]] — majority vote of K nearest
- [[decision_trees]] — yes/no question tree
- [[support_vector_machines]] — maximum margin boundary
- [[confusion_matrix]] — TP, FP, TN, FN
- [[evaluation_metrics_classification]] — accuracy, precision, recall, F1

### Clustering (Course 01)
- [[k_means]] — K centroids, iterative
- [[hierarchical_clustering]] — tree of nested clusters
- [[dbscan]] — density-based, arbitrary shape
- [[hdbscan]] — hierarchical DBSCAN, varying density
- [[elbow_method]] — choosing optimal K

### Dimensionality Reduction (Course 01)
- [[pca]] — linear, preserves global variance
- [[tsne]] — non-linear, for visualisation only
- [[umap]] — non-linear, faster than t-SNE, preserves more

### Recommender Systems (Course 01)
- [[content_based_filtering]] — item features
- [[collaborative_filtering]] — user similarity
- [[cold_start_problem]] — no history = no recommendation

### Deep Learning Foundations (Courses 02–05)
- [[neural_network]] — layers of connected neurons
- [[activation_functions]] — sigmoid, ReLU, tanh, softmax
- [[backpropagation]] — how networks learn from errors
- [[gradient_descent]] — minimising the loss function
- [[vanishing_gradient]] — why deep nets were hard to train
- [[dropout]] — regularisation for neural networks
- [[batch_normalisation]] — stabilising training
- [[weight_initialisation]] — starting point matters
- [[seq2seq]] — sequence-to-sequence learning, variable-length input/output
- [[encoder_decoder]] — compress input → generate output, teacher forcing
- [[self_attention]] — Q/K/V projections, scaled dot-product, the core of Transformers

### Keras & TensorFlow (Courses 02–03)
- [[keras]] — high-level API for building neural nets
- [[tensorflow]] — backend framework
- [[one_hot_encoding]] — convert integer labels to binary vectors
- [[softmax]] — output activation for multi-class classification
- [[categorical_crossentropy]] — loss function for classification
- [[adam_optimizer]] — adaptive learning rate optimiser
- [[model_saving_keras]] — save/load .keras models with full optimizer state
- [[keras_functional_api]] — graph-based model building; multi-input, skip connections, shared layers
- [[custom_keras_layers]] — subclass Layer; build/call pattern; add_weight; trainable parameters
- [[weight_initialisation]] — random_normal, glorot_uniform, He init; breaking symmetry
- [[data_augmentation]] — ImageDataGenerator; rotation/zoom/flip/shear; preprocessing_function; implicit regularisation
- [[image_normalisation]] — feature-wise (global μ/σ) vs sample-wise (per-image AGC); datagen.fit()
- [[optimizers]] — SGD, RMSprop, Adam; update rules; adaptive vs fixed LR; when to use each
- [[cnn]] — convolutional neural networks (images)
- [[conv2d]] — convolutional layer: filters, kernels, feature maps, strides
- [[max_pooling]] — spatial downsampling, translation invariance
- [[flatten]] — bridge from 3D feature maps to Dense classifier head
- [[rnn]] — recurrent neural networks (sequences)
- [[autoencoder]] — encoder-decoder reconstruction; target=input; bottleneck; MSE loss; anomaly detection
- [[transpose_convolution]] — Conv2DTranspose; learned upsampling; stride-based spatial expansion; GANs, autoencoders, segmentation
- [[gan]] — generative adversarial networks
- [[transfer_learning]] — feature extraction vs. fine-tuning, frozen weights, pretrained models
- [[vgg16]] — 16-layer CNN, imagenet weights, 7×7×512 feature map
- [[binary_crossentropy]] — loss for sigmoid/2-class output, complement to categorical_crossentropy
- [[blip_image_captioning]] — vision-language model, caption & summarize images via Hugging Face

### PyTorch (Courses 04–05)
- [[pytorch]] — flexible deep learning framework
- [[tensor]] — PyTorch's core data structure
- [[autograd]] — automatic differentiation
- [[dataloader]] — batching and shuffling data
- [[softmax_regression]] — multi-class classification

### GenAI & LLMs (Courses 07–11)
- [[transformer_architecture]] — attention is all you need
- [[attention_mechanism]] — how transformers focus
- [[positional_encoding]] — order in sequences
- [[gpt]] — decoder-based language model
- [[bert]] — encoder-based language model
- [[embeddings]] — dense vector representations of text
- [[word2vec]] — CBOW and Skip-gram
- [[tokenisation]] — text to numbers
- [[fine_tuning]] — adapting pretrained models
- [[peft]] — parameter-efficient fine-tuning
- [[lora]] — low-rank adaptation
- [[qlora]] — quantised LoRA
- [[rlhf]] — reinforcement learning from human feedback
- [[dpo]] — direct preference optimisation
- [[ppo]] — proximal policy optimisation
- [[vae]] — variational autoencoders
- [[diffusion_models]] — generative models for images

### AI Agents & RAG (Courses 12–13)
- [[rag]] — retrieval augmented generation
- [[langchain]] — framework for LLM apps
- [[vector_database]] — storing and querying embeddings
- [[prompt_engineering]] — designing effective prompts
- [[ai_agents]] — autonomous LLM-based systems
- [[gradio]] — quick UI for ML models

---

## 🛠️ Tools & Libraries

| Tool | Purpose | Introduced in |
|---|---|---|
| `scikit-learn` | Classical ML | [[01_ML_with_Python]] |
| `pandas` | Data manipulation | [[01_ML_with_Python]] |
| `numpy` | Numerical arrays | [[01_ML_with_Python]] |
| `matplotlib` / `seaborn` | Visualisation | [[01_ML_with_Python]] |
| `Keras` | Neural networks (high-level) | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] |
| `TensorFlow` | Deep learning backend | [[03_Deep_Learning_Keras_TensorFlow]] |
| `PyTorch` | Flexible deep learning | [[04_Intro_Neural_Networks_PyTorch]] |
| `Hugging Face` | Pretrained models & fine-tuning | [[09_GenAI_Language_Modeling_Transformers]] |
| `LangChain` | LLM apps & agents | [[12_AI_Agents_RAG_LangChain]] |
| `Gradio` | UI for ML models | [[13_Project_GenAI_Apps_RAG_LangChain]] |
| `NLTK / spaCy` | NLP preprocessing | [[07_GenAI_LLMs_Architecture_Data_Prep]] |
| `Vector DB` (ChromaDB etc.) | Embeddings storage | [[12_AI_Agents_RAG_LangChain]] |

---

## 💼 Portfolio Tracker

| Course | Project idea | Status | GitHub |
|---|---|---|---|
| [[01_ML_with_Python]] | ML classifier on telecom/RF dataset | ⬜ To build | |
| [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | ANN on a real-world dataset | 🔄 In progress | |
| [[03_Deep_Learning_Keras_TensorFlow]] | CNN image classifier or autoencoder | ⬜ To build | |
| [[04_Intro_Neural_Networks_PyTorch]] | PyTorch regression / classification model | ⬜ To build | |
| [[05_Deep_Learning_PyTorch]] | Deep CNN with transfer learning | ⬜ To build | |
| [[06_AI_Capstone_Deep_Learning]] | Full DL pipeline — image classification | ⬜ To build | |
| [[09_GenAI_Language_Modeling_Transformers]] | Fine-tuned transformer for NLP task | ⬜ To build | |
| [[13_Project_GenAI_Apps_RAG_LangChain]] | RAG-powered QA bot with Gradio UI | ⬜ To build | |

> **Your edge:** connect projects to your telecom/RF/nanotech background — it makes your portfolio distinctive. Example: RF signal classification with CNNs, network anomaly detection with autoencoders, telecom QA bot with RAG.

---

## ❓ Open Questions

- [ ] How exactly does [[gradient_descent]] update weights in [[backpropagation]]?
- [ ] When do I choose [[k_nearest_neighbors]] vs [[decision_trees]] vs [[support_vector_machines]]?
- [ ] What is the real difference between [[ridge_lasso_regression]] and standard regression?
- [ ] How does [[attention_mechanism]] handle long sequences without losing context?
- [ ] What is the difference between [[gpt]] (decoder) and [[bert]] (encoder) in practice?
- [ ] How does [[rlhf]] connect to [[reinforcement_learning]] from Course 01?

---

## 🔗 Related MOCs

- [[Deep_Learning_MOC]] ← create when you reach [[03_Deep_Learning_Keras_TensorFlow]]
- [[GenAI_MOC]] ← create when you reach [[07_GenAI_LLMs_Architecture_Data_Prep]]

---

## 📅 Study Log

| Week | Course | Focus | Notes |
|---|---|---|---|
| 2026-W20 | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | ANN forward propagation, activation functions, backpropagation XOR, Keras regression | |
| 2026-W20 | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | Keras classification — MNIST, one-hot encoding, Softmax, categorical cross-entropy, Adam, model save/load | Keras_Classification_MNIST_v1.ipynb |
| 2026-W20 | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | Convolutional Neural Networks — Conv2D, MaxPooling, Flatten, feature maps, 1-block vs 2-block CNN, batch size vs epoch trade-off | DL0101EN-4-1-Convolutional-Neural-Networks-with-K.ipynb |
| 2026-W21 | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | Transformers with Keras — self-attention (Q/K/V), encoder-decoder seq2seq, teacher forcing, Glorot vs He initialisation, Adam vs Adagrad | DL0101EN-4-1-Transformers-with-Keras-py-v1.ipynb |
| 2026-W22 | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | Final project: VGG16 transfer learning for aircraft damage classification (dent vs crack), BLIP image captioning & summarization via custom Keras layer | Final_Project_Classification_and_Captioning.ipynb |
| 2026-W23 | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | Keras Functional API — Input/Dense/Model wiring, compile/train/evaluate, Dropout, BatchNormalization, activation functions (ReLU/Tanh/Sigmoid) | Implementing the Functional API in Keras.ipynb |
| 2026-W23 | [[02_Intro_Deep_Learning_Neural_Networks_Keras]] | ✅ Module complete | |
| 2026-W23 | [[03_Deep_Learning_Keras_TensorFlow]] | 🔄 Module started | |
| 2026-W25 | [[03_Deep_Learning_Keras_TensorFlow]] | Custom Keras layers — `Layer` subclassing, `build()`/`call()`, `add_weight()`, weight init, Dropout, model visualization | Creating_Custom_Layers_and_Models.ipynb |
| 2026-W25 | [[03_Deep_Learning_Keras_TensorFlow]] | Data augmentation — `ImageDataGenerator`, geometric transforms, feature-wise/sample-wise normalisation, custom `preprocessing_function` (AWGN) | Advanced_Data_Augmentation_with_Keras.ipynb |
| 2026-W25 | [[03_Deep_Learning_Keras_TensorFlow]] | Transfer learning — VGG16 feature extraction, layer freezing/unfreezing, fine-tuning (last 4 layers), SGD vs RMSprop vs Adam comparison | Transfer_Learning_Implementation.ipynb |
| 2026-W25 | [[03_Deep_Learning_Keras_TensorFlow]] | Transpose convolution — Conv2DTranspose, encoder-decoder reconstruction, MSE loss, kernel size / Dropout / activation experiments | Practical_Application_of_Transpose_Convolution.ipynb |
| | | | |

---

> **How to use this MOC:**
> - Click any `[[concept]]` or `[[folder]]` to open or create that note
> - Update ✅/🔄/⬜ status as you progress through each course
> - Add open questions as you encounter them
> - Add GitHub links as you push portfolio projects
> - Graph view in Obsidian shows this MOC as the central hub of your AI knowledge graph

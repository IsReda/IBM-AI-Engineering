---
course: IBM AI Engineering
module: 01_ML_with_Python
status: ✅ Done
created: 2026-05-14
tags: [ml, scikit-learn, supervised, unsupervised, course/IBM-AI-Engineering]
---

# Course 1 — Machine Learning with Python

**Course MOC:** [[AI_ML_MOC]] · **Next module:** [[02_Deep_Learning_Keras]]
**Visual reference:** [[IBM_Course1_StudyNotes.html|🎨 Original HTML version]]

---

## 01 · What is Machine Learning?

> **Feynman check** — "We show the computer thousands of examples. It finds the pattern. Then it can guess on data it has never seen — without us writing the rules."

### Diagram — AI, ML & Deep Learning

```mermaid
graph TD
    AI["🤖 **Artificial Intelligence**
    Any technique to mimic human behavior
    ─────────────────────────────
    Expert systems · Robotics"]

    ML["📊 **Machine Learning**
    Learns from data without explicit rules
    ─────────────────────────────
    SVM · Trees · Regression"]

    DL["🧠 **Deep Learning**
    Neural networks with many layers
    ─────────────────────────────
    Images · Speech · Language"]

    AI --> ML
    ML --> DL

    style AI fill:#ede9fb,stroke:#6b4fbb,stroke-width:2px,color:#3b2a8a
    style ML fill:#e0f5f0,stroke:#0f7c6b,stroke-width:2px,color:#064e43
    style DL fill:#fff3e0,stroke:#b06000,stroke-width:2px,color:#7a4200
```

### Diagram — 3 Types of Machine Learning

```mermaid
graph LR
    ML["🎯 Machine Learning"]

    SUP["🏷️ **Supervised**
    Input → Known Output
    Labeled data
    ───────────────
    🏠 House price
    📧 Spam filter
    🩺 Disease diagnosis"]

    UNS["🔍 **Unsupervised**
    Input → Find Patterns
    No labels needed
    ───────────────
    👥 Customer segments
    🔎 Anomaly detection
    📰 Topic modelling"]

    REI["🎮 **Reinforcement**
    Action → Reward/Penalty
    Trial & error
    ───────────────
    🎮 Game playing AI
    🤖 Robotics
    📈 Trading bots"]

    ML --> SUP
    ML --> UNS
    ML --> REI

    style SUP fill:#ede9fb,stroke:#6b4fbb,stroke-width:2px,color:#3b2a8a
    style UNS fill:#e0f5f0,stroke:#0f7c6b,stroke-width:2px,color:#064e43
    style REI fill:#fff3e0,stroke:#b06000,stroke-width:2px,color:#7a4200
```

### Diagram — The ML Pipeline

```mermaid
graph LR
    A["📥 Raw Data
    CSV · DB · API"]
    B["🧹 Clean Data
    Handle nulls"]
    C["⚙️ Features
    Engineering"]
    D["🎓 Train Model
    Fit on data"]
    E["📊 Evaluate
    Metrics · Test"]
    F["🚀 Deploy
    API · App"]

    A --> B --> C --> D --> E --> F

    style A fill:#ede9fb,stroke:#6b4fbb
    style B fill:#e0f5f0,stroke:#0f7c6b
    style C fill:#fff3e0,stroke:#b06000
    style D fill:#fdecea,stroke:#c0392b
    style E fill:#e8f1fc,stroke:#1a5fa8
    style F fill:#e8f5e9,stroke:#2e7d32
```

### Key Vocabulary

| Term | Definition |
|---|---|
| `Feature` | An input variable (column) — e.g. house size, age |
| `Label / Target` | The output we want to predict — e.g. price, category |
| `Training set` | Data used to fit the model (~70–80%) |
| `Test set` | Held-out data to evaluate performance (~20–30%) |
| `Overfitting` | Model memorises training data, fails on new data |
| `Underfitting` | Model too simple, misses the real pattern |

---

## 02 · Regression

Predicts a **continuous numerical value** — price, temperature, salary, score.

### Diagram — Linear Regression

```mermaid
graph LR
    X["📐 Input features
    e.g. house size (m²)"]
    M["📈 Linear Model
    y = b₀ + b₁x
    intercept + slope × x"]
    Y["💰 Predicted value
    e.g. price in €k"]

    X --> M --> Y

    style X fill:#ede9fb,stroke:#6b4fbb,color:#3b2a8a
    style M fill:#e8f1fc,stroke:#1a5fa8,color:#0d3d6e
    style Y fill:#e8f5e9,stroke:#2e7d32,color:#1b4d1e
```

> **Intuition** — fits a straight line through a scatter of points, minimising the distance between the line and each data point. More features = multiple linear regression, more dimensions, same idea.

### Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| `MAE` | mean(\|actual − predicted\|) | Average absolute error, easy to interpret |
| `MSE` | mean((actual − predicted)²) | Penalises large errors heavily |
| `RMSE` | √MSE | Same units as the target variable |
| `R²` | 1 − SS_res / SS_tot | % of variance explained. 1 = perfect, 0 = useless |

### Code — scikit-learn

```python
# scikit-learn — Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R² = {r2_score(y_test, y_pred):.3f}")
```

---

## 03 · Classification

Predicts a **discrete category** — spam/not spam, disease/healthy, cat/dog.

### Diagram — Decision Tree (simplified)

```mermaid
graph TD
    ROOT["❓ Is email size > 5kb?
    Root node"]

    N1["❓ Has suspicious links?
    Internal node"]

    N2["❓ Sender known?
    Internal node"]

    L1["✅ Not Spam"]
    L2["🚫 Spam"]
    L3["✅ Not Spam"]
    L4["🚫 Spam"]

    ROOT -->|"No"| N1
    ROOT -->|"Yes"| N2
    N1 -->|"No"| L1
    N1 -->|"Yes"| L2
    N2 -->|"Yes"| L3
    N2 -->|"No"| L4

    style ROOT fill:#ede9fb,stroke:#6b4fbb,color:#3b2a8a
    style N1 fill:#e0f5f0,stroke:#0f7c6b,color:#064e43
    style N2 fill:#fff3e0,stroke:#b06000,color:#7a4200
    style L1 fill:#e8f5e9,stroke:#2e7d32,color:#1b4d1e
    style L2 fill:#fdecea,stroke:#c0392b,color:#7b2418
    style L3 fill:#e8f5e9,stroke:#2e7d32,color:#1b4d1e
    style L4 fill:#fdecea,stroke:#c0392b,color:#7b2418
```

### Diagram — Confusion Matrix

```mermaid
quadrantChart
    title Confusion Matrix
    x-axis "Predicted Negative" --> "Predicted Positive"
    y-axis "Actual Negative" --> "Actual Positive"
    quadrant-1 "TP ✅ True Positive
    Correctly predicted sick"
    quadrant-2 "FN ⚠️ False Negative
    Missed sick patient"
    quadrant-3 "TN ✅ True Negative
    Correctly predicted healthy"
    quadrant-4 "FP ⚠️ False Positive
    False alarm"
```

> **Key insight** — Accuracy is misleading on imbalanced classes. A model that always predicts "healthy" gets 99% accuracy if only 1% of patients are sick — but catches zero actual cases.

### Classification metrics

| Metric | Formula | When to use |
|---|---|---|
| `Accuracy` | Correct / Total | Balanced classes only |
| `Precision` | TP / (TP + FP) | When false alarms are costly |
| `Recall` | TP / (TP + FN) | When missing a case is costly (e.g. medical) |
| `F1 Score` | 2 × (P × R) / (P + R) | Balance between precision and recall |

### Algorithms covered

| Algorithm | How it works | Wikilink |
|---|---|---|
| KNN | Majority vote of K nearest points | [[k_nearest_neighbors]] |
| Decision Tree | Series of yes/no questions | [[decision_trees]] |
| Logistic Regression | Outputs probability (despite the name) | [[logistic_regression]] |
| SVM | Finds optimal separating boundary | [[support_vector_machines]] |

---

## 04 · Clustering

**Unsupervised** — groups similar data with no labels. Algorithm finds structure on its own.

### Diagram — K-Means Clustering (K=3)

```mermaid
graph TD
    START["🎲 Place K random centroids"]
    ASSIGN["📍 Assign each point
    to nearest centroid"]
    MOVE["📐 Move centroid to
    mean of its cluster"]
    CHECK{"Any centroid moved?"}
    DONE["✅ Done — K clusters found"]

    START --> ASSIGN
    ASSIGN --> MOVE
    MOVE --> CHECK
    CHECK -->|"Yes"| ASSIGN
    CHECK -->|"No"| DONE

    style START fill:#ede9fb,stroke:#6b4fbb,color:#3b2a8a
    style ASSIGN fill:#e0f5f0,stroke:#0f7c6b,color:#064e43
    style MOVE fill:#fff3e0,stroke:#b06000,color:#7a4200
    style CHECK fill:#e8f1fc,stroke:#1a5fa8,color:#0d3d6e
    style DONE fill:#e8f5e9,stroke:#2e7d32,color:#1b4d1e
```

> **Choosing K** — use the **elbow method**: plot inertia (within-cluster distance) vs K. The "elbow" point where the curve flattens = optimal K. Beyond it, more clusters give diminishing returns.

### Algorithms compared

| Algorithm | Shape of clusters | Needs K? | Handles noise? |
|---|---|---|---|
| [[k_means]] | Spherical | Yes | No |
| [[hierarchical_clustering]] | Any (tree) | No | Partly |
| [[dbscan]] | Arbitrary | No | Yes |

---

## 05 · Recommender Systems

Suggest items to users based on patterns.

### Diagram — Content-Based vs Collaborative Filtering

```mermaid
graph LR
    subgraph CB ["🎨 Content-Based Filtering"]
        U1["👤 User"]
        I1["🎬 Inception
        (liked)"]
        R1["🎬 Interstellar
        🎬 The Matrix"]
        U1 -->|"liked sci-fi thriller"| I1
        I1 -->|"similar genre/style"| R1
    end

    subgraph CF ["👥 Collaborative Filtering"]
        U2["👤 You"]
        U3["👤 Similar
        user"]
        I2["🎬 Film D
        (not seen)"]
        U2 ---|"similar taste"| U3
        U3 -->|"rated highly"| I2
        I2 -->|"recommend"| U2
    end

    style CB fill:#ede9fb,stroke:#6b4fbb,color:#3b2a8a
    style CF fill:#e0f5f0,stroke:#0f7c6b,color:#064e43
```

> **Cold start problem** — new users or items have no history, so collaborative filtering can't recommend anything. Workaround: use content-based filtering until enough data is gathered.

| Approach | Based on | Limitation |
|---|---|---|
| Content-based | Item features | Misses serendipity, stays in a "bubble" |
| Collaborative filtering | User similarity | Cold start problem |
| Hybrid | Both | More complex but more robust |

---

## 🧠 Self-Quiz — Test Yourself

> Try answering out loud before reading the answers. Feynman method.

**Q1 · What is the difference between [[supervised_learning]] and [[unsupervised_learning]]?**
> Supervised uses labeled data — the model learns from known input→output pairs. Unsupervised uses unlabeled data — the model finds hidden patterns with no predefined answers.

**Q2 · R² = 0.85 — what does this mean?**
> The model explains 85% of the variance in the target variable. The remaining 15% is unexplained noise or factors not captured by the features.

**Q3 · Medical test for a rare disease — Precision or Recall matters most? Why?**
> Recall matters most. Missing a real sick patient (false negative) is more dangerous than a false alarm. You'd rather over-alert and investigate than miss a case.

**Q4 · What happens in K-Means if you choose K too large?**
> The model overfits — clusters become too granular and stop representing meaningful groups. Points may end up in their own cluster, losing all insight.

**Q5 · What is the "cold start problem" in recommenders?**
> When a new user or item has no history, collaborative filtering cannot make recommendations. Content-based filtering is a common workaround — it uses item features, not user history.

**Q6 · Name 3 signs your model is overfitting.**
> ① Very high training accuracy but low test accuracy. ② Model performs perfectly on known data but poorly on new samples. ③ Overly complex model with too many parameters relative to dataset size.

**Q7 · What does the elbow method tell you?**
> It helps choose the optimal K in clustering. Plot inertia (within-cluster distance) vs K — pick the point where the curve bends. Beyond this, more clusters give diminishing returns.

---

## 📝 Confusion Log

*Things that confused me and how I resolved them — add entries here.*

| What confused me | How I resolved it | Date |
|---|---|---|
| | | |

---

## Open questions to revisit

- How does [[gradient_descent]] (preview from [[02_Deep_Learning_Keras]]) connect to fitting these regression models?
- When exactly does [[polynomial_regression]] start overfitting?

---

## Concept wikilinks in this note

[[supervised_learning]] · [[unsupervised_learning]] · [[reinforcement_learning]]
[[linear_regression]] · [[logistic_regression]] · [[decision_trees]]
[[k_nearest_neighbors]] · [[support_vector_machines]]
[[k_means]] · [[hierarchical_clustering]] · [[dbscan]]
[[confusion_matrix]] · [[gradient_descent]] · [[neural_network]]

---

## Source materials

- Coursera: IBM AI Engineering — Course 1
- Local folder: `learning/IBM_AI_Engineering/01_ML_with_Python/`
- Projects: `01_ML_with_Python/04_Projects/`

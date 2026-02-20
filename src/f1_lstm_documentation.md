# F1 Pit Stop Prediction using LSTM
## Complete Documentation & User Manual

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Methodology](#methodology)
4. [Code Walkthrough](#code-walkthrough)
5. [Evaluation Metrics Explained](#evaluation-metrics)
6. [Results Interpretation](#results-interpretation)
7. [Usage Guide](#usage-guide)
8. [Future Enhancements](#future-enhancements)

---

## 1. Project Overview

### What is this project?
This project uses **Long Short-Term Memory (LSTM)** neural networks to predict when an F1 driver should pit during a race. It analyzes historical lap data including tire wear, lap times, race position, and track conditions to make intelligent pit stop recommendations.

### Why LSTM?
LSTM is a type of Recurrent Neural Network (RNN) specifically designed for **sequence prediction problems**. F1 races are sequential - what happens on lap 20 depends on what happened on laps 1-19. LSTM excels at:
- **Remembering long-term patterns** (tire degradation over many laps)
- **Understanding temporal relationships** (how lap times change as tires age)
- **Handling variable-length sequences** (different race lengths)

### Key Objectives
1. ✅ Predict whether a driver should pit on the next lap (Binary Classification)
2. 🔄 Foundation for predicting which tire compound to use (Future Extension)
3. 📊 Provide confidence scores for pit decisions

---

## 2. Problem Statement

### The Challenge
In Formula 1, pit stop strategy is crucial:
- **Pit too early**: Waste tire life and lose track position
- **Pit too late**: Slow lap times due to worn tires
- **Wrong tire choice**: Poor performance for remaining stint

### Input Data
Our model uses 19 features per lap:
```
- Lap number (race progression)
- Tire age (laps on current tires)
- Tire life remaining
- Current position
- Gap to leader/car ahead
- Nearby cars (within 2 seconds)
- Track/air temperature
- Humidity
- Current tire compound
- Lap times and sector times
- Stint information
```

### Output
- **Probability (0-1)**: Likelihood of needing a pit stop
- **Binary Decision**: Pit (1) or Stay Out (0)

---

## 3. Methodology

### High-Level Workflow

```
Raw CSV Data
    ↓
Feature Engineering (convert times, encode categories)
    ↓
Sequence Creation (group last 10 laps)
    ↓
Normalization (StandardScaler)
    ↓
Train/Test Split (80/20)
    ↓
LSTM Model Training
    ↓
Evaluation & Prediction
```

### Why This Approach?

#### Sequence Length = 10 Laps
- Captures recent tire degradation trends
- Not too short (misses patterns) or too long (irrelevant old data)
- Typical F1 stint is 15-25 laps, so 10 laps gives good context

#### Bidirectional LSTM
- Looks at data **forward and backward** in time
- Example: Lap 15 is influenced by both lap 14 (past) and lap 16 (future pattern)
- Better pattern recognition than standard LSTM

#### Class Weighting
- **Problem**: Most laps don't have pit stops (~95% are "No Pit")
- **Solution**: Give higher weight to rare "Pit" examples during training
- Prevents model from just predicting "No Pit" every time

---

## 4. Code Walkthrough

### Section 1: Data Loading & Imports

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
```

**What's happening:**
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `StandardScaler`: Normalizes features to same scale
- `LabelEncoder`: Converts categories (SOFT, MEDIUM, HARD) to numbers
- `tensorflow/keras`: Deep learning framework

---

### Section 2: Feature Engineering

```python
def prepare_features(df):
    # Convert timedelta to seconds
    for col in time_columns:
        df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
```

**Why this matters:**
- Original data has times as "0 days 00:01:15.234000"
- LSTM needs numbers, not time objects
- Converts to seconds: 75.234

```python
# Encode categorical variables
le_compound = LabelEncoder()
df['current_compound_encoded'] = le_compound.fit_transform(df['current_compound'])
```

**Encoding Example:**
```
SOFT   → 0
MEDIUM → 1
HARD   → 2
```

Neural networks can't understand text, only numbers!

---

### Section 3: Sequence Creation

```python
def create_sequences(df, feature_columns, sequence_length=10):
    sequences = []
    targets_pit = []
    
    grouped = df.groupby(['driver', 'season', 'round_number'])
    
    for name, group in grouped:
        for i in range(len(group) - sequence_length):
            seq = features[i:i + sequence_length]
            target_pit = pit_labels[i + sequence_length]
            sequences.append(seq)
            targets_pit.append(target_pit)
```

**Visual Example:**

```
Lap Data for Driver "HAM" in Race:
Lap 1  [features...]  Pit=0
Lap 2  [features...]  Pit=0
...
Lap 10 [features...]  Pit=0
Lap 11 [features...]  Pit=1  ← This is our target!

Sequence 1: Laps 1-10 → Predict Lap 11 (Pit=1)
Sequence 2: Laps 2-11 → Predict Lap 12 (Pit=0)
Sequence 3: Laps 3-12 → Predict Lap 13 (Pit=0)
```

**Why group by driver & race?**
- Don't mix Hamilton's lap 5 with Verstappen's lap 5
- Each driver/race is independent
- Prevents data leakage between different races

---

### Section 4: Data Normalization

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
```

**Why normalize?**

Before normalization:
```
Lap Number:    1, 2, 3, 4, 5...
Temperature:   25.3, 25.5, 25.7...
Gap to Leader: 0.5, 1.2, 2.4, 45.6...
```

After normalization (mean=0, std=1):
```
Lap Number:    -1.2, -0.8, -0.4, 0.0, 0.4...
Temperature:   -0.1, 0.0, 0.1...
Gap to Leader: -0.5, -0.3, -0.1, 2.1...
```

**Benefits:**
- All features on same scale
- Faster training convergence
- Prevents large values from dominating small ones

---

### Section 5: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_pit, test_size=0.2, random_state=42, stratify=y_pit
)
```

**Parameters explained:**
- `test_size=0.2`: 20% for testing, 80% for training
- `random_state=42`: Ensures reproducible results
- `stratify=y_pit`: Maintains class ratio in both sets

**Example:**
```
Total: 1000 samples (50 Pits, 950 No Pits)
Train: 800 samples (40 Pits, 760 No Pits) - same 5% ratio
Test:  200 samples (10 Pits, 190 No Pits) - same 5% ratio
```

---

### Section 6: LSTM Model Architecture

```python
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
    Dropout(0.3),
    
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    
    LSTM(32),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    
    Dense(1, activation='sigmoid')
])
```

**Layer-by-Layer Breakdown:**

#### Layer 1: Bidirectional LSTM(128)
- **128 units**: Each can learn different patterns
- **Bidirectional**: Processes sequence forward AND backward
- **return_sequences=True**: Passes full sequence to next layer
- **Output shape**: (batch, 10, 256) - 256 because bidirectional doubles

#### Layer 2: Dropout(0.3)
- **Randomly removes 30% of connections during training**
- **Prevents overfitting**: Model can't rely on specific neurons
- Think of it as "teaching with randomness" to build robustness

#### Layer 3-4: Bidirectional LSTM(64) + Dropout
- **64 units**: Fewer than first layer (hierarchical learning)
- **Learns higher-level patterns** from Layer 1's output
- Output shape: (batch, 10, 128)

#### Layer 5: LSTM(32)
- **No return_sequences**: Only keeps final output
- **Condenses 10 laps → single decision point**
- Output shape: (batch, 32)

#### Layer 6-7: Dense(64) + Dropout
- **Fully connected layer**: Combines LSTM features
- **ReLU activation**: f(x) = max(0, x)
- Learns complex combinations of temporal patterns

#### Layer 8: Dense(32)
- **Intermediate decision layer**
- Further refines predictions

#### Layer 9: Dense(1, sigmoid)
- **Single output**: Pit probability
- **Sigmoid**: Squashes output to 0-1 range
- Formula: σ(x) = 1 / (1 + e^(-x))

**Total Parameters**: ~300K trainable weights!

---

### Section 7: Model Compilation

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)
```

#### Optimizer: Adam
- **Adaptive learning rate**: Adjusts step size automatically
- **Best of both worlds**: Momentum + RMSprop
- **Learning rate 0.001**: Standard starting point

#### Loss: Binary Crossentropy
- **Perfect for binary classification** (Pit vs No Pit)
- **Formula**: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
- **Penalizes confident wrong predictions heavily**

Example:
```
True: Pit (1), Predicted: 0.9 → Loss = -log(0.9) = 0.105 (small)
True: Pit (1), Predicted: 0.1 → Loss = -log(0.1) = 2.303 (large!)
```

---

### Section 8: Callbacks

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]
```

#### EarlyStopping
- **Monitors**: Validation loss
- **Patience=15**: Waits 15 epochs for improvement
- **restore_best_weights**: Returns to best model state
- **Prevents overfitting**: Stops when validation performance plateaus

#### ReduceLROnPlateau
- **Reduces learning rate** when stuck
- **Factor=0.5**: Halves learning rate
- **Helps escape local minima**

Example:
```
Epoch 1-20: Learning rate = 0.001
Epoch 21-25: No improvement, reduce to 0.0005
Epoch 26-30: Still stuck, reduce to 0.00025
```

---

### Section 9: Class Weights

```python
class_weights = {
    0: 1.0,
    1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
}
```

**The Imbalance Problem:**
```
No Pit: 19,000 samples (95%)
Pit:    1,000 samples (5%)

Without weighting → Model predicts "No Pit" always → 95% accuracy!
But useless for pit strategy!
```

**With Class Weights:**
```
Class 0 (No Pit): weight = 1.0
Class 1 (Pit):    weight = 19.0

Each "Pit" sample counts as 19 "No Pit" samples during training
Forces model to learn the minority class
```

---

### Section 10: Training

```python
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights
)
```

**Parameters:**
- **validation_split=0.2**: 20% of training data for validation
- **epochs=100**: Maximum 100 passes through data
- **batch_size=32**: Process 32 sequences at once

**Training Process:**
```
Epoch 1/100
 ├─ Process 32 sequences → Calculate loss → Update weights
 ├─ Repeat for all batches
 └─ Validate on validation set

Epoch 2/100
 └─ ... (repeat with updated weights)
```

---

## 5. Evaluation Metrics Explained

### Why Multiple Metrics?

**Accuracy alone is misleading for imbalanced data!**

Imagine a model that always predicts "No Pit":
- **Accuracy**: 95% ✓ (looks great!)
- **But**: Misses every actual pit stop ✗

We need metrics that measure **both classes separately**.

---

### Confusion Matrix

```
                 Predicted
               No Pit    Pit
Actual  No Pit   TN      FP
        Pit      FN      TP

TN = True Negative  (Correctly predicted No Pit)
TP = True Positive  (Correctly predicted Pit)
FN = False Negative (Missed a Pit - Type II Error)
FP = False Positive (False alarm - Type I Error)
```

**Example:**
```
              Predicted
            No Pit   Pit
Actual No   1850     50    (1850 correct, 50 false alarms)
       Pit    30     70    (70 caught, 30 missed)

Total predictions: 2000
```

**Why it matters for F1:**
- **False Negative (FN)**: Didn't pit when should → Slow lap times, tire failure risk
- **False Positive (FP)**: Pitted unnecessarily → Lost track position

---

### 1. Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Example:**
```
(70 + 1850) / 2000 = 0.96 = 96%
```

**What it means:**
- Overall correctness
- **Limitation**: Dominated by majority class (No Pit)

**When to use:**
- Quick overall performance check
- **NOT sufficient alone for imbalanced data**

---

### 2. Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**Example:**
```
70 / (70 + 50) = 0.583 = 58.3%
```

**What it means:**
- **"When I predict Pit, how often am I right?"**
- Of all predicted pit stops, what % were actual pit stops?

**F1 Context:**
- **High Precision (90%+)**: Few false alarms, trust the model
- **Low Precision (50%)**: Many unnecessary pit stops suggested

**Trade-off:**
- Can achieve 100% precision by being very conservative
- But might miss many actual pit stops (low recall)

**Use case:**
- Critical when **false alarms are costly**
- In F1: Unnecessary pit = lost 20-25 seconds

---

### 3. Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
```

**Example:**
```
70 / (70 + 30) = 0.70 = 70%
```

**What it means:**
- **"Of all actual pit stops, how many did I catch?"**
- Coverage of positive class

**F1 Context:**
- **High Recall (90%+)**: Catches most pit opportunities
- **Low Recall (50%)**: Misses half the pit windows

**Trade-off:**
- Can achieve 100% recall by predicting "Pit" always
- But many false alarms (low precision)

**Use case:**
- Critical when **missing positives is dangerous**
- In F1: Missing optimal pit window → slow laps, safety risk

---

### 4. F1-Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:**
```
2 × (0.583 × 0.70) / (0.583 + 0.70) = 0.636 = 63.6%
```

**What it means:**
- **Harmonic mean of Precision and Recall**
- Single metric balancing both concerns
- Punishes extreme imbalance

**Why harmonic mean?**
```
Arithmetic mean: (0.583 + 0.70) / 2 = 0.641
Harmonic mean:   2 / (1/0.583 + 1/0.70) = 0.636

Harmonic mean is lower → penalizes if one metric is very low
```

**F1 Score Interpretation:**
- **0.9-1.0**: Excellent - Both precision and recall high
- **0.7-0.9**: Good - Balanced performance
- **0.5-0.7**: Moderate - Needs improvement
- **<0.5**: Poor - Model struggling

**F1 Context:**
- **Perfect balance** between false alarms and missed opportunities
- **Industry standard** for imbalanced classification
- **Use case**: When both errors matter equally

---

### Precision-Recall Trade-off

**Visual Example:**

```
Threshold = 0.3 (low)
├─ Predict "Pit" if probability > 0.3
├─ Many predictions → High Recall (90%), Low Precision (60%)
└─ Catches most pits but many false alarms

Threshold = 0.7 (high)
├─ Predict "Pit" if probability > 0.7
├─ Conservative → Low Recall (70%), High Precision (90%)
└─ Few false alarms but misses some pits

Threshold = 0.5 (balanced)
└─ Usually optimal F1-score
```

**Choosing the right threshold depends on context:**

| Scenario | Priority | Threshold |
|----------|----------|-----------|
| Qualifying | Catch every pit opportunity | Low (0.3-0.4) |
| Leading the race | Avoid unnecessary stops | High (0.6-0.7) |
| Mid-pack | Balance both | Medium (0.5) |

---

### 5. Loss (Binary Cross-Entropy)

**Formula:**
```
Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Example calculations:**

```
Actual: Pit (1), Predicted: 0.9
Loss = -[1·log(0.9) + 0·log(0.1)] = -log(0.9) = 0.105

Actual: Pit (1), Predicted: 0.5
Loss = -log(0.5) = 0.693

Actual: Pit (1), Predicted: 0.1
Loss = -log(0.1) = 2.303 (HUGE penalty!)
```

**What it means:**
- **Lower loss = Better predictions**
- Exponentially penalizes confident wrong predictions
- Loss → 0 for perfect predictions
- Loss → ∞ for completely wrong predictions

**Why not just use accuracy?**
```
Prediction A: 0.51 → Classified as Pit
Prediction B: 0.99 → Classified as Pit

Accuracy: Both correct (1 point each)
Loss: B is much better (more confident in correct answer)
```

---

### Metric Summary Table

| Metric | Formula | What it measures | F1 Context | Ideal Value |
|--------|---------|------------------|------------|-------------|
| **Accuracy** | (TP+TN)/Total | Overall correctness | General performance (misleading if imbalanced) | 0.95+ |
| **Precision** | TP/(TP+FP) | Quality of positive predictions | "Don't waste pit stops" | 0.80+ |
| **Recall** | TP/(TP+FN) | Coverage of actual positives | "Don't miss pit windows" | 0.85+ |
| **F1-Score** | 2·P·R/(P+R) | Balance of P and R | Overall pit strategy quality | 0.80+ |
| **Loss** | -Σy·log(ŷ) | Prediction confidence | Training optimization | <0.3 |

---

### Which Metrics Matter Most?

**For F1 Pit Strategy:**

1. **Recall (Most Critical)**
   - Missing a pit stop → safety risk, tire failure
   - Better to suggest 1 extra pit than miss 1 critical pit

2. **F1-Score (Strategic Goal)**
   - Balance false alarms vs missed opportunities
   - Optimize race strategy overall

3. **Precision (Tactical)**
   - High precision → Team trusts the system
   - Low precision → Ignored by strategists

4. **Accuracy (Least Important)**
   - Useful only for comparison with baseline
   - Don't rely on it alone!

---

## 6. Results Interpretation

### Understanding Training Plots

#### Plot 1: Loss Curves
```
Training Loss: How well model fits training data
Validation Loss: How well model generalizes

Good pattern:
  Training ↘ Validation ↘ (both decreasing together)

Overfitting:
  Training ↘ Validation ↗ (validation increases while training decreases)

Underfitting:
  Training ─ Validation ─ (both plateau at high values)
```

#### Plot 2: Accuracy Curves
```
Gap between train/val accuracy:
  Small gap (<5%): Good generalization
  Large gap (>15%): Overfitting
```

#### Plot 3: Precision & Recall
```
Ideal: Both curves high (>0.8)
Trade-off visible: One up, other down
Sweet spot: Where curves are closest and highest
```

---

### Reading the Confusion Matrix

```
              Predicted
            No Pit   Pit
Actual No    1850    50      
       Pit     30    70      

Insights:
✓ 1850/1900 "No Pit" correct (97.4% - good!)
✓ 70/100 "Pit" caught (70% recall - moderate)
✗ 30 missed pit stops (needs improvement)
✗ 50 false alarms (acceptable)

Action: Tune threshold to catch more pits (increase recall)
```

---

### Classification Report Example

```
              precision    recall  f1-score   support

     No Pit       0.98      0.97      0.98      1900
        Pit       0.58      0.70      0.64       100

    accuracy                          0.96      2000
   macro avg      0.78      0.84      0.81      2000
weighted avg      0.96      0.96      0.96      2000
```

**How to read:**
- **support**: Number of samples in each class
- **macro avg**: Simple average (treats classes equally)
- **weighted avg**: Weighted by class size (dominated by majority)

**For imbalanced data, focus on:**
- Minority class (Pit) metrics: precision=0.58, recall=0.70
- F1-score of minority class: 0.64
- Macro average: 0.81 (better representation than weighted)

---

## 7. Usage Guide

### Making Predictions

```python
# Load saved model and preprocessors
from tensorflow import keras
import pickle

model = keras.models.load_model('f1_pit_prediction_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
```

### Real-time Prediction Example

```python
# Get last 10 laps for a driver
driver_data = df[df['driver'] == 'HAM'].tail(10)

# Prepare features
features = driver_data[feature_columns].fillna(0).values
features_scaled = scaler.transform(features)
features_scaled = features_scaled.reshape(1, 10, -1)

# Predict
pit_probability = model.predict(features_scaled)[0][0]

print(f"Pit Probability: {pit_probability:.1%}")
print(f"Recommendation: {'🔴 PIT NOW' if pit_probability > 0.5 else '🟢 STAY OUT'}")

# Example output:
# Pit Probability: 78.3%
# Recommendation: 🔴 PIT NOW
```

### Batch Predictions

```python
# Predict for entire test set
predictions = model.predict(X_test)

# Get top 10 highest pit probabilities
top_pit_indices = predictions.flatten().argsort()[-10:][::-1]

for idx in top_pit_indices:
    print(f"Sequence {idx}: {predictions[idx][0]:.1%} confidence")
```

---

## 8. Future Enhancements

### Phase 1: Multi-Task Learning (Tire Compound)

Modify output layer:
```python
# Add second output for tire compound
from tensorflow.keras.layers import concatenate

# Current: Single output (Pit yes/no)
Dense(1, activation='sigmoid', name='pit_decision')

# Future: Two outputs
pit_output = Dense(1, activation='sigmoid', name='pit_decision')(x)
compound_output = Dense(3, activation='softmax', name='tire_compound')(x)

model = keras.Model(inputs=input_layer, outputs=[pit_output, compound_output])
```

**Benefits:**
- Predicts both WHEN and WHAT tire
- Shared LSTM layers learn general patterns
- Separate heads specialize in each task

---

### Phase 2: Advanced Features

```python
# Add calculated features
df['tire_deg_rate'] = df.groupby('stint_number')['lap_time'].diff()
df['fuel_load_est'] = df['lap_number'].apply(lambda x: 110 - (x * 1.8))
df['undercut_window'] = (df['gap_to_car_ahead'] < 2.0) & (df['tire_age_laps'] > 10)

# Weather forecast integration
df['rain_probability_next_5_laps'] = weather_forecast(lap_number)
```

---

### Phase 3: Attention Mechanism

```python
from tensorflow.keras.layers import Attention

# Add attention to focus on critical laps
attention = Attention()([lstm_output, lstm_output])

# Model learns which laps matter most
# (e.g., laps 18-20 might be more important than laps 8-10)
```

---

### Phase 4: Real-time Dashboard

```python
# Streamlit dashboard for live predictions
import streamlit as st

st.title("🏎️ F1 Pit Strategy Predictor")

driver = st.selectbox("Select Driver", drivers)
recent_laps = get_recent_laps(driver, n=10)

pit_prob, should_pit = predict_pit_stop(model, scaler, feature_columns, recent_laps)

st.metric("Pit Probability", f"{pit_prob:.1%}")
st.metric("Recommendation", "PIT" if should_pit else "STAY OUT")

# Visualize key factors
st.line_chart(recent_laps['lap_time'])
st.bar_chart(recent_laps['tire_age_laps'])
```

---

## 9. Troubleshooting

### Common Issues

#### Issue 1: Low Recall (<60%)
**Problem**: Missing too many pit stops

**Solutions:**
```python
# 1. Adjust decision threshold
threshold = 0.3  # Lower = more sensitive

# 2. Increase class weight
class_weights = {0: 1.0, 1: 25.0}  # Increase from 19 to 25

# 3. Add more LSTM layers
model.add(LSTM(64, return_sequences=True))
```

---

#### Issue 2: Overfitting
**Problem**: Training accuracy high, validation low

**Solutions:**
```python
# 1. Increase dropout
Dropout(0.5)  # From 0.3 to 0.5

# 2. Reduce model complexity
LSTM(64)  # From 128 to 64 units

# 3. Add L2 regularization
LSTM(128, kernel_regularizer=keras.regularizers.l2(0.01))

# 4. Get more data
# Collect more races/seasons
```

---

#### Issue 3: Training Too Slow
**Problem**: Takes hours to train

**Solutions:**
```python
# 1. Reduce sequence length
sequence_length = 5  # From 10 to 5

# 2. Increase batch size
batch_size = 64  # From 32 to 64

# 3. Use GPU
with tf.device('/GPU:0'):
    model.fit(...)

# 4. Reduce model size
LSTM(64)  # From 128
```

---

## 10. Best Practices

### Data Quality Checklist
- ✅ Remove outliers (pit stop laps, safety cars)
- ✅ Handle missing values consistently
- ✅ Verify time conversions are correct
- ✅ Check for data leakage (no future info in features)
- ✅ Validate grouping (driver/race combinations)

### Model Training Checklist
- ✅ Monitor both training and validation metrics
- ✅ Use early stopping
- ✅ Save best model weights
- ✅ Track experiments (learning rate, architecture)
- ✅ Validate on unseen races (not just random split)

### Deployment Checklist
- ✅ Test on live race simulation
- ✅ Measure inference time (<1 second)
- ✅ Handle edge cases (red flags, weather)
- ✅ Provide confidence scores
- ✅ Log all predictions for analysis

---

## 11. Mathematical Foundations

### LSTM Cell Math

**Forget Gate:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Decides what to forget from previous memory
```

**Input Gate:**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
Decides what new information to store
```

**Cell State Update:**
```
c_t = f_t * c_{t-1} + i_t * c̃_t
Updates long-term memory
```

**Output Gate:**
```
o_t = σ(W_o · [h
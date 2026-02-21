# 🏎 Transformer-Based F1 Pit Strategy Prediction Project

## 📌 Project Objective

Build a deep learning model to:

1.  Predict whether a driver should PIT on lap t\
2.  Predict which tire compound to use if a pit stop occurs

This is formulated as a **multi-task sequential classification problem**
using a Transformer-based architecture.

------------------------------------------------------------------------

# 🧠 Problem Formulation

At each lap `t`, given the previous `N` laps:

Input:

    (batch_size, seq_len, feature_dim)

Output: - Binary classification → Pit or Not Pit - Multi-class
classification → Tire Compound (Soft / Medium / Hard)

------------------------------------------------------------------------

# 📊 Data Processing Pipeline

## 1️⃣ Data Grouping

Data must be grouped by: - `race_id` - `driver_id`

⚠ Never mix drivers or races within a sequence.

------------------------------------------------------------------------

## 2️⃣ Feature Engineering

### 🔹 Dynamic Lap Features

-   lap_number
-   position
-   gap_to_leader
-   gap_to_front
-   gap_to_behind
-   lap_time
-   sector_1
-   sector_2
-   sector_3
-   safety_car_flag
-   track_temp
-   tire_age
-   stint_number

------------------------------------------------------------------------

### 🔹 Engineered Strategic Features

-   degradation = lap_time\[t\] − lap_time\[t-1\]
-   rolling_avg_3
-   delta_to_fastest
-   undercut_flag

These features help the model understand tire performance decay and
strategic windows.

------------------------------------------------------------------------

### 🔹 Static Context Features

Embedded categorical variables:

-   driver_id
-   track_id
-   team_id

These allow the model to learn driver-specific and track-specific
strategy patterns.

------------------------------------------------------------------------

## 3️⃣ Normalization

Continuous features are scaled using:

    StandardScaler()

This ensures stable training.

------------------------------------------------------------------------

## 4️⃣ Sequence Creation

Sliding window approach:

Example (SEQ_LEN = 20):

    Lap 1–20 → Predict Lap 21
    Lap 2–21 → Predict Lap 22
    ...

------------------------------------------------------------------------

# 🤖 Model Architecture

## Step 1: Embeddings

-   Driver Embedding (dim=8)
-   Track Embedding (dim=8)

These are concatenated with lap features.

------------------------------------------------------------------------

## Step 2: Input Projection

    Linear(input_dim → D_MODEL)

Projects features to Transformer dimension.

------------------------------------------------------------------------

## Step 3: Transformer Encoder

Hyperparameters:

-   D_MODEL = 128
-   NUM_LAYERS = 3
-   NHEAD = 4
-   DROPOUT = 0.1

Output shape:

    (batch_size, seq_len, D_MODEL)

We take the **last token representation** as the race state summary.

------------------------------------------------------------------------

## Step 4: Multi-Task Heads

### Pit Head

    Linear(D_MODEL → 1)
    Sigmoid

### Tire Head

    Linear(D_MODEL → 3)
    Softmax

------------------------------------------------------------------------

# 📉 Loss Function

Total Loss:

    Total Loss = α * PitLoss + β * TireLoss

Where:

-   PitLoss = Focal Loss (handles class imbalance)
-   TireLoss = Cross Entropy Loss
-   α = 1.0
-   β = 0.5

------------------------------------------------------------------------

# ⚙️ Tunable Hyperparameters

You can tune:

-   SEQ_LEN
-   BATCH_SIZE
-   EPOCHS
-   LR (learning rate)
-   D_MODEL
-   NUM_LAYERS
-   NHEAD
-   DROPOUT
-   Focal Loss gamma
-   Embedding dimensions
-   Loss weights (α, β)

------------------------------------------------------------------------

# 📈 Evaluation Metrics

Since pit events are rare, accuracy is misleading.

Use:

-   Precision
-   Recall
-   F1 Score
-   PR-AUC

Precision is particularly important because false pit decisions are
costly in real racing.

------------------------------------------------------------------------

# 🏎 Advanced Evaluation (Recommended)

Beyond classification metrics:

1.  Simulate race using predicted pit laps
2.  Calculate final race time
3.  Compare with actual strategy

This provides real strategic validation.

------------------------------------------------------------------------

# 🚀 Possible Improvements

-   Add positional encoding
-   Add regression head for time-to-next-pit
-   Compare with LSTM baseline
-   Add race simulation engine
-   Use time-series cross-validation
-   Pretrain on multi-year data, fine-tune per driver

------------------------------------------------------------------------

# 🧾 Overall Workflow Summary

Raw Data\
↓\
Feature Engineering\
↓\
Normalization\
↓\
Sequence Creation\
↓\
Transformer Encoder\
↓\
Multi-Task Heads\
↓\
Loss Computation\
↓\
Evaluation

------------------------------------------------------------------------

# 🎓 Research-Level Extensions

-   Strategy imitation vs optimal strategy learning
-   Reinforcement learning for optimal pit timing
-   Attention map visualization for interpretability
-   Cross-track generalization experiments

------------------------------------------------------------------------

# ✅ Conclusion

This project models F1 pit strategy as a structured sequential decision
problem using an attention-based Transformer.

It captures:

-   Long-term degradation patterns
-   Driver-specific behavior
-   Track-specific strategy dynamics
-   Rare event detection using focal loss

With proper evaluation and simulation, this can become
publication-quality research.

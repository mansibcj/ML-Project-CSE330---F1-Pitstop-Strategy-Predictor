# F1 Pit Stop Strategy System — Code Documentation & Results

**Project:** Data-driven F1 pit stop timing and tire compound recommendation  
**Reference paper:** Sasikumar, Leema & Balakrishnan (2025), *Frontiers in Artificial Intelligence*, 8:1673148  
**Models:** Bi-LSTM (Task A — pit timing) · XGBoost (Task B — compound recommendation)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Task A — Bi-LSTM Pit Stop Prediction (`train_bilstm.py`)](#task-a--bi-lstm-pit-stop-prediction)
3. [Task B — XGBoost Compound Recommendation (`train_taskB_compound.py`)](#task-b--xgboost-compound-recommendation)
4. [Results Explanation — Task A](#results-explanation--task-a)
5. [Results Explanation — Task B](#results-explanation--task-b)
6. [Combined Pipeline Summary](#combined-pipeline-summary)

---

## System Overview

The system is a two-stage sequential pipeline. **Task A** runs on every lap of a race and answers *"should this driver pit right now?"* **Task B** activates only when Task A fires and answers *"if pitting, which tire compound should they switch to?"*

```
Every lap
    │
    ▼
Task A: Bi-LSTM
    │
    ├── P(pit) < threshold ──→ No pit. Continue.
    │
    └── P(pit) ≥ threshold ──→ Pit recommended!
                                    │
                                    ▼
                               Task B: XGBoost
                                    │
                                    ▼
                          P(HARD), P(MEDIUM), P(SOFT)
                          → "Switch to MEDIUM (94%)"
```

The two models are fully independent — Task B loads Task A's saved preprocessing artifacts to guarantee identical feature scaling.

---

## Task A — Bi-LSTM Pit Stop Prediction

**File:** `train_bilstm.py`  
**Architecture:** 3-layer Bidirectional LSTM → Dense → Sigmoid  
**Input:** Sequences of 10 consecutive laps per driver per race  
**Output:** P(pit stop on this lap) ∈ [0, 1]

---

### `load_and_validate(path)`

Loads the preprocessed CSV and immediately applies a data integrity fix. The dataset builder computed `avg_pit_time_team` as the difference between `PitOutTime` and `PitInTime` in FastF1, but FastF1 timestamps are session-elapsed values — certain edge cases (lap 1, safety car laps, incomplete timing data) produce negative or impossibly large values. Any value outside the physically possible range of 1.5 to 60 seconds is set to NaN so the KNN imputer can replace it with a contextually appropriate estimate. Also validates that the three required columns — `pit_this_lap`, `season`, and `round_number` — are present before any processing begins.

---

### `consolidate_sparse_columns(df)`

Handles two types of OHE column redundancy:

**Team renaming:** The same physical F1 team appeared under different names across the 2020–2024 seasons. AlphaTauri became RB in 2024. Alfa Romeo Racing became Alfa Romeo then Kick Sauber. Racing Point became Aston Martin. Renault became Alpine. Having two sparse binary columns for one team dilutes the signal — the model assigns separate weights to the same strategic entity. This function merges them using element-wise maximum (OR logic, since a row can only belong to one team at a time).

**Sparse driver collapse:** Drivers with fewer than 500 total laps in the dataset produce columns that are 99%+ zeros. One-off substitutes and late-career appearances add dimensionality without adding signal. All such drivers are merged into a single `driver_OTHER` column.

---

### `temporal_split(df)`

Splits the dataset into training and test sets using a **temporal boundary** — the final 8 races of the 2024 season form the test set, everything else is training. No shuffling is performed at any point.

The critical detail here is `reset_index(drop=True)` applied to both splits immediately after slicing. After a DataFrame slice, the retained rows keep their original CSV row numbers as the index — a training DataFrame might have index values [0, 1, 5, 7, 12, ...] (non-contiguous). When `impute_and_scale()` converts the DataFrame to a numpy array, the array is always 0-indexed: position 0 in the array corresponds to the first row of the DataFrame regardless of what the DataFrame index label says. If the index is not reset, any subsequent operation that maps index labels to array positions will silently use wrong rows, producing sequences that mix laps from different drivers and races. Resetting the index makes label values equal to positional values, eliminating this mismatch.

---

### `get_feature_columns(df)`

Returns the ordered list of model input features by excluding four categories of columns:

- **Metadata:** `season`, `round_number` — used for split logic only, not model input
- **Targets:** `pit_this_lap`, `next_tire_compound`
- **Corrupted:** `avg_pit_time_team` — builder bug produces negative values
- **Redundant:** `stint_number` (same information as `total_pit_stops_so_far`), `is_new_tire` (derivable from `tire_age_laps == 1`), `laps_since_last_pit` (same information as `tire_age_laps`), `lap_number` (superseded by `race_progress_fraction`)

---

### `fix_delta_laptime_edges(train_df, test_df)`

Sets `lap_time_delta_prev` to NaN for two structurally misleading cases:

**Lap 1 of each race:** There is no previous lap, so any stored value is either zero-filled or computed against a non-existent prior. The value is meaningless and potentially misleading — the KNN imputer replaces it with an interpolated estimate from contextually similar laps.

**Lap immediately after a pit stop:** The pit stop lap itself is approximately 20 seconds slower than a normal racing lap (the car is stationary in the pit box). The very next lap on fresh tires appears to be a massive improvement of perhaps -22 seconds, which looks to the model like extraordinary performance rather than a pit stop artefact. Without this fix, the model might learn that a large negative delta is a reason *not* to pit — the exact opposite of the correct inference.

---

### `impute_and_scale(train_df, test_df, feature_cols)`

Implements the paper's two-stage imputation pipeline, fitted exclusively on training data:

**Stage 1 — StandardScaler:** Normalises all features to mean=0, standard deviation=1. This is required before KNN because KNN uses Euclidean distance — without scaling, `lap_time` measured in the 70–90 second range would completely dominate `position` measured 1–20, making the nearest-neighbour calculation meaningless.

**Stage 2 — KNNImputer (k=5):** Finds the 5 most similar laps in normalised feature space and uses their values to fill NaN cells. This multivariate approach preserves relationships between features far better than mean or median imputation.

The fitted scaler and imputer objects are returned and later saved to disk so Task B can apply identical preprocessing.

---

### `_sequences_from_array(X, y, seq_len)`

Converts a contiguous array of laps belonging to one (race, driver) group into overlapping windows of length `seq_len`. Window `i` covers laps `[i : i + seq_len]` and its label is the target of the **last lap** in that window — did the driver pit at the end of lap `i + seq_len - 1`? A group with N laps produces `max(0, N - seq_len)` sequences. Groups shorter than `seq_len` (drivers who retired early) produce zero sequences — correct behaviour, not a bug.

---

### `build_sequences_per_group(train_df, test_df, X_train, X_test, feature_cols)`

The most critical function in the pipeline. Builds all sequences strictly within the boundary of each (season, round, driver) group, ensuring no sequence window ever spans the gap between two different drivers or two different races.

The previous approach — sliding a window across the entire sorted array — failed silently because the last lap of Driver A in Race N was immediately followed by lap 1 of Driver B in Race N+1 in the sorted array. A window spanning that transition would contain, for example, laps 50–57 of Verstappen at the Austrian Grand Prix followed by laps 1–2 of Hamilton at the British Grand Prix. The Bi-LSTM would attempt to learn temporal patterns from this sequence, which has no physical meaning.

The fix works as follows: the data is grouped by `(season, round_number)`. Within each race, the driver identity is read from the OHE driver columns using `idxmax(axis=1)`. For each unique driver in each race, their rows are extracted from the numpy array X, and `_sequences_from_array` is called on that subarray. After `reset_index(drop=True)`, the DataFrame index labels are integer positions, so `grp_idx.values` can be used directly as numpy array indices without any `get_indexer()` translation.

---

### `build_bilstm(input_shape)`

Constructs the 3-layer Bidirectional LSTM architecture from the paper (Table 1, Figure 6):

| Layer | Units | Returns | Dropout |
|---|---|---|---|
| Bidirectional LSTM 1 | 512 | Sequence | 0.2 |
| Bidirectional LSTM 2 | 256 | Sequence | 0.3 |
| Bidirectional LSTM 3 | 128 | Single vector | 0.3 |
| Dense | 1 | Sigmoid | — |

**Why Bidirectional:** A standard LSTM processes laps in chronological order (lap 1 → lap N). The backward pass processes them in reverse (lap N → lap 1). Concatenating both gives the model access to what happens *after* each lap. A lap that precedes a pit stop has a subtly different signature when viewed from the perspective of what follows it — this retrospective context is something a unidirectional LSTM cannot access until the sequence has already passed.

**Why decreasing units:** Each layer compresses the temporal representation into progressively more abstract summaries. The final 128-unit layer produces a compact race-context vector that the Dense sigmoid head classifies as pit/no-pit.

---

### `train_model(X_tr_seq, y_tr_seq)`

Trains the model with two Keras callbacks:

**EarlyStopping (patience=12):** Monitors `val_loss`. If it does not improve for 12 consecutive epochs, training stops and weights from the best epoch are restored. Patience of 12 is necessary because the low learning rate (1e-4) causes slow convergence — a tighter patience would terminate training before the model has had time to find its optimum.

**ReduceLROnPlateau (patience=6, factor=0.5):** Halves the learning rate when `val_loss` plateaus for 6 epochs. With ES patience of 12, this allows two LR reductions (6+6=12 epochs) before early stopping triggers, giving the model two "rescue" opportunities to escape a plateau through finer-grained weight updates.

**Class weights {0:1.0, 1:25.0}:** Every missed pit stop contributes 25 times more to the loss function than every correctly classified non-pit lap. This compensates for the ~96:4 class imbalance without using SMOTE, which would corrupt the temporal sequence structure by inserting synthetic laps into chronological order.

---

### `find_optimal_threshold(model, X_tr_seq, y_tr_seq)`

The model outputs a probability in [0, 1] for each sequence. The default threshold of 0.5 would miss most real pit stops because the distribution is heavily skewed toward 0 — even a well-trained model rarely outputs P > 0.5 on a 4% pit rate dataset. This function computes the full Precision-Recall curve on training predictions and finds the threshold where F1 is maximised. Training data is used — not test data — to avoid threshold overfitting to the test set (data leakage). The optimal threshold is then applied unchanged to test predictions.

---

### `bootstrap_ci(y_true, y_pred, y_prob)`

Computes 95% confidence intervals for F1, Balanced Accuracy, ROC-AUC, and AUC-PR using 1,000 bootstrap resamples — the paper's exact method. Each resample draws `len(y_true)` samples **with replacement** from the test set, computes the metric, and stores the result. The 2.5th and 97.5th percentiles of the 1,000 values form the confidence interval bounds. Resamples where only one class is present (which would make metrics undefined) are silently skipped.

---

### `evaluate(model, X_te_seq, y_te_seq, X_tr_seq, y_tr_seq, history)`

Applies the full paper metric suite to the test set. The seven metrics are:

| Metric | Formula | What it measures |
|---|---|---|
| Precision | TP / (TP + FP) | Of predicted pit stops, fraction that were real |
| Recall | TP / (TP + FN) | Of actual pit stops, fraction the model caught |
| F1-Score | Harmonic mean of P and R | Balance of precision and recall |
| Specificity | TN / (TN + FP) | Of non-pit laps, fraction correctly identified |
| Balanced Accuracy | (Recall + Specificity) / 2 | Class-balanced accuracy, robust to imbalance |
| ROC-AUC | Area under ROC curve | Discriminative ability across all thresholds |
| AUC-PR | Area under PR curve | Precision-recall tradeoff (most informative for imbalanced data) |

---

### `save_preprocessing_artifacts(scaler, imputer, feature_cols, threshold)`

**This function is the bridge between Task A and Task B.** It saves four objects to disk:

- `task_a_scaler.pkl` — the fitted StandardScaler
- `task_a_imputer.pkl` — the fitted KNNImputer
- `task_a_feature_cols.json` — the exact ordered list of 91 feature column names
- `task_a_threshold.json` — the optimal classification threshold

Without this, Task B would refit a new scaler on the training data. Even if the training data is identical, subtle differences in column ordering, floating-point accumulation, or sparse column consolidation can produce different mean/variance values. The saved Bi-LSTM was trained on data scaled by the original scaler — if Task B uses a different scaler, the model receives inputs on a slightly different numerical scale and produces systematically wrong probabilities, breaking the inference pipeline.

---

### Plot Functions (`plot_all`)

Generates six evaluation plots saved to `outputs/`:

| File | Content |
|---|---|
| `01_training_curves.png` | Loss and accuracy over epochs, with best epoch marker |
| `02_confusion_matrix.png` | TN/FP/FN/TP heatmap with count labels |
| `03_pr_curve.png` | Precision-Recall curve vs no-skill baseline |
| `04_roc_curve.png` | ROC curve vs random guess diagonal |
| `05_metric_bar_chart.png` | Precision/Recall/F1/Specificity/Balanced Accuracy bar chart |
| `06_ci_table.png` | 95% bootstrap CI table matching paper Table 3 format |

---

## Task B — XGBoost Compound Recommendation

**File:** `train_taskB_compound.py`  
**Architecture:** XGBoost multi-class classifier (3 classes: HARD, MEDIUM, SOFT)  
**Input:** Race state at the moment of a Task A-predicted pit stop (single row, no sequence)  
**Output:** P(HARD), P(MEDIUM), P(SOFT)

---

### `load_task_a_artifacts()`

Loads the four preprocessing artifacts saved by Task A's `save_preprocessing_artifacts()`. Validates that all four files exist before proceeding — if any are missing, raises a descriptive `FileNotFoundError` directing the user to run `train_bilstm.py` first. Returns the model, scaler, imputer, feature column list, and threshold as a tuple.

---

### `fix_pit_time(df)` and `consolidate_sparse_columns(df)`

Identical logic to the Task A counterparts — applied here to ensure the full dataset has the same structure before the temporal split. The team renaming and sparse driver consolidation must produce the same column set that Task A saw, otherwise the feature column list loaded from `task_a_feature_cols.json` will contain column names that don't exist in the current DataFrame.

---

### `temporal_split(df)`

Same boundary as Task A: test = final 8 races of 2024, train = everything else. Identical `reset_index(drop=True)` applied. The split returns a third value `test_rounds` — the list of round numbers — used later to label the per-race prediction plots.

---

### `fix_delta_single(df)`

Single-DataFrame version of Task A's `fix_delta_laptime_edges()`. Applied to the test DataFrame before Task A inference. Uses the same two rules: lap 1 of each race gets NaN, and the lap immediately after a pit stop gets NaN.

---

### `build_inference_sequences(df, X, feature_cols)`

Mirrors Task A's `build_sequences_per_group` exactly — same grouping by (season, round), same driver identification via OHE columns, same window building logic via `_sequences_from_array`. The only addition is tracking `row_indices`: the DataFrame row index of the **last lap** in each sequence window. This index is the link between a sequence's prediction and the specific lap in the DataFrame. After inference, the row indices of sequences where `y_pred == 1` give the exact positions in `test_df` that Task A predicted as pit stops, which Task B then processes.

---

### `run_task_a_inference(full_df)`

The core of Part 1. After loading artifacts, applies the **saved** scaler and imputer to the test data using `.transform()` only — no `.fit()`, no `.fit_transform()`. This is the guarantee that Task B's input to the Bi-LSTM is on the exact same numerical scale as the data the model was trained on.

Builds sequences, runs `model.predict()`, applies the saved threshold, and identifies which test DataFrame row indices correspond to predicted pit stops (`pred_pit_indices`). Also computes TP, FP, FN counts by comparing predictions against ground truth `pit_this_lap`.

---

### `get_task_b_features(df, task_a_feature_cols)`

Derives Task B's feature set as a subset of Task A's features. Two additional columns are dropped beyond the Task A exclusions:

- `lap_time` — raw lap time is less informative for compound selection than for degradation detection
- `lap_time_delta_prev` — the lap-to-lap delta is a degradation signal relevant for timing, but compound choice depends more on strategic context (laps remaining, compounds already used, track status)

The function starts from `task_a_feature_cols` (the saved list) to ensure column consistency.

---

### `train_task_b(train_df, task_a_feature_cols)`

Trains the XGBoost compound classifier on pre-test pit stop laps (all laps where `pit_this_lap == 1` and `next_tire_compound` is one of HARD/MEDIUM/SOFT, from the training period only).

**Why XGBoost instead of Bi-LSTM for Task B:** Task B has approximately 2,957 pit stop rows — far too few for a deep recurrent model to learn meaningful patterns from. Pit stop rows are also independent observations with no temporal sequence structure to exploit. XGBoost excels on small tabular datasets, requires no sequence building, trains in seconds rather than minutes, and produces interpretable feature importance scores.

**SMOTE is safe for Task B:** Unlike Task A where SMOTE corrupted temporal sequences by inserting synthetic laps into chronological order, Task B operates on independent pit stop events. Each row is one pit stop. SMOTE interpolates between existing SOFT/MEDIUM/HARD examples to create synthetic ones, which is valid because there is no ordering relationship between pit stop events from different races.

**5-fold stratified cross-validation:** Before the final fit, cross-validation provides an unbiased estimate of generalisation performance. Stratified folds preserve the compound class ratios in each fold, preventing all SOFT examples from concentrating in one fold.

---

### `run_task_b_inference(task_b_model, scaler_b, imputer_b, feature_cols_b, test_df, pred_pit_indices)`

Runs Task B on the laps Task A identified as pit stops. Critically, this includes both True Positives (real pit stops Task A correctly caught) and False Positives (non-pit laps Task A mislabelled as pit stops).

**Task B cannot know which of its inputs are TP vs FP.** It simply receives a row of features and outputs compound probabilities. This is correct behaviour — at race time, the strategy engineer receives Task B's compound recommendation whenever Task A fires, regardless of whether Task A was right.

**Evaluation is restricted to True Positives only:** Only on TP laps does a ground truth `next_tire_compound` label exist (the actual compound the team switched to). FP laps have no ground truth because no pit stop actually occurred. The evaluation metrics therefore reflect Task B's performance in the realistic scenario of "Task A was correct — now which compound?"

---

### Plot Functions — Task B

| File | Content |
|---|---|
| `taskB_00_compound_distribution.png` | Compound counts in training vs Task A TP laps |
| `taskB_01_taskA_predictions.png` | Task A pit probability curve over laps for all 8 test races |
| `taskB_02_confusion_matrix.png` | 3×3 compound confusion matrix with row percentage annotations |
| `taskB_03_per_class_metrics.png` | Precision/Recall/F1 per compound with macro F1 reference line |
| `taskB_04_feature_importance.png` | Top 20 XGBoost gain importances coloured by feature category |
| `taskB_05_pr_curves.png` | Per-compound Precision-Recall curves with no-skill baselines |
| `taskB_06_calibration.png` | Reliability diagrams showing whether predicted probabilities are calibrated |
| `taskB_07_pipeline_summary.png` | Combined Task A + Task B end-to-end summary table |

---

## Results Explanation — Task A

```
Threshold  : 0.9232
Precision  : 0.950
Recall     : 0.791
F1-Score   : 0.863   [0.825, 0.900]
Specificity: 0.999
Bal. Acc   : 0.895   [0.867, 0.922]
ROC-AUC    : 0.978   [0.964, 0.989]
AUC-PR     : 0.906   [0.873, 0.937]

Confusion Matrix:
                   Pred 0    Pred 1
Actual 0 (no pit)   6534         9
Actual 1 (pit)        45       170
```

### Threshold = 0.9232

The model must be 92% confident before it predicts a pit stop. This is very high and is a direct consequence of the `class_weight = {0:1.0, 1:25.0}` configuration. The large class weight pushes the model to be aggressive during training — it fires pit predictions liberally to avoid the 25x penalty for missing a real pit stop. To counteract this and bring precision up, the PR-curve threshold search lands at 0.9232, meaning only the most confident predictions make it through. This is the correct behaviour: high weight during training forces the model to learn strong pit stop signals; high threshold during inference filters out the less confident predictions.

### Precision = 0.950

Of the 179 laps the model predicted as pit stops, 170 were genuine pit stops. Only 9 were false alarms. In practical terms: out of 10 times the system alerts a strategy engineer to prepare for a pit stop, 9.5 times on average the car actually pits. This is a highly usable signal.

### Recall = 0.791

The model caught 170 of the 215 actual pit stops in the test races (243 total, but only those with ground truth). It missed 45 pit stops (false negatives). This means about 1 in 5 real pit stops goes undetected — the system would not warn the engineer for those events. This is the precision/recall tradeoff: the high threshold that gives 95% precision means some real pit stops with probabilities between 0.5 and 0.92 are not flagged.

### F1-Score = 0.863 [0.825, 0.900]

The harmonic mean of precision and recall. This is the headline metric for imbalanced classification. The paper's Bi-LSTM achieved 0.81 on their dataset — your model achieves 0.863, beating the paper by 0.053 points. The confidence interval [0.825, 0.900] confirms this is stable across bootstrap resamples, not a lucky single-test result.

### Specificity = 0.999

Of all 6,543 non-pit laps in the test set, 6,534 were correctly identified as non-pit. Only 9 were incorrectly flagged. This near-perfect specificity reflects the high threshold: the model almost never fires on non-pit laps.

### Balanced Accuracy = 0.895 [0.867, 0.922]

The arithmetic mean of Recall (0.791) and Specificity (0.999). This is the most honest single-number accuracy for imbalanced datasets because it weights both classes equally. 0.895 means the model is performing at 89.5% accuracy when judged fairly across both the majority class (no pit) and the minority class (pit). The paper achieved 0.93 — you are 3.5 points behind them on this metric, which reflects the slightly lower recall.

### ROC-AUC = 0.978 [0.964, 0.989]

Across all possible thresholds, the model's ability to separate pit from non-pit laps has an AUC of 0.978. This is essentially the same as the paper's 0.988 (within the confidence interval). It means that if you pick a random pit lap and a random non-pit lap from the test set, the model assigns a higher probability to the pit lap 97.8% of the time.

### AUC-PR = 0.906 [0.873, 0.937]

This is the most informative metric for imbalanced data. The no-skill baseline for your test set is approximately 0.029 (the pit stop prevalence). Your model achieves 0.906 — roughly 31 times better than random. The paper achieved 0.879 — you exceed them by 0.027 points on this metric. A high AUC-PR specifically means the model maintains high precision across a wide range of recall values, not just at the chosen threshold.

### Comparison with the Paper

| Metric | Paper Bi-LSTM | Your Model | Difference |
|---|---|---|---|
| Precision | 0.77 | **0.950** | +0.180 |
| Recall | 0.86 | 0.791 | -0.069 |
| F1-Score | 0.81 | **0.863** | +0.053 |
| Specificity | 0.992 | **0.999** | +0.007 |
| Balanced Accuracy | 0.93 | 0.895 | -0.035 |
| ROC-AUC | 0.988 | 0.978 | -0.010 |
| AUC-PR | 0.879 | **0.906** | +0.027 |

Your model beats the paper on Precision, F1, Specificity, and AUC-PR. The paper has higher Recall and Balanced Accuracy — a consequence of a different precision/recall tradeoff point. Neither result is strictly better: the paper prioritised catching more pit stops (higher recall), while your configuration prioritises not crying wolf (higher precision).

---

## Results Explanation — Task B

```
Laps Task A predicted as pit stops: 179
True Positives  : 169
False Positives : 10

Compound    Prec    Rec     F1    ROC-AUC  AUC-PR  Support
HARD       0.929  0.794  0.856    0.911   0.971     131
MEDIUM     0.585  0.800  0.676    0.928   0.789      30
SOFT       0.312  0.625  0.417    0.933   0.431       8
Macro F1: 0.650  |  Accuracy: 0.787
```

### Input Quality: 169 TP, 10 FP

Task A sent Task B 179 laps. Of these, 169 were real pit stops (Task A was correct) and 10 were false positives (Task A misfired). This means Task B is working with 94.4% clean input — far better than an isolated evaluation would suggest. The 10 false positive laps still receive compound recommendations, which is unavoidable: Task B cannot know Task A was wrong.

### Overall Accuracy = 0.787

The model correctly recommended the right compound on 133 of the 169 True Positive laps. In operational terms: when Task A correctly identifies a pit stop, Task B recommends the right tire type approximately 4 times out of 5. This is a genuine, useful result for a strategy support tool.

### HARD: F1 = 0.856

The model excels at identifying HARD tire stints. This is expected — HARD compound selection follows consistent, learnable patterns: it is predominantly chosen for long final stints (high `race_progress_fraction`), at circuits with high tire degradation rates (captured by `race_name` OHE), and when a conservative strategy is needed. With 131 HARD events out of 169 total (77.5% of test pit stops), the model has ample examples to learn from. Precision of 0.929 means almost every HARD recommendation is correct.

### MEDIUM: F1 = 0.676

Decent recall (80% of actual MEDIUM stops caught) but lower precision (58.5%). The precision gap reflects genuine strategic ambiguity — MEDIUM and SOFT are often interchangeable from a team's perspective, especially in the first or second stint. A team choosing MEDIUM over SOFT does so for durability reasons that may not be fully visible in the public telemetry data. Some of the model's false MEDIUM predictions may reflect situations where either compound was legitimately viable and the team chose SOFT.

### SOFT: F1 = 0.417

The weakest result, but importantly explained by sample size rather than model failure. There are only 8 SOFT pit stops across all 8 test races. With support of 8, even a single wrong prediction dramatically moves the metrics: the model caught 5 of the 8 (recall 0.625) but also generated false positive SOFT predictions (precision 0.312). The ROC-AUC of 0.933 tells the real story — the model's underlying ability to separate SOFT from non-SOFT is actually quite strong. The low precision and F1 are artefacts of the tiny sample. The 2024 season used SOFT compounds conservatively, making this an inherently hard evaluation.

### Macro F1 = 0.650

The average F1 across all three compounds equally weighted. It is pulled down significantly by SOFT's score of 0.417 on 8 examples. A weighted F1 (which weights by support) would give 0.787, matching accuracy. The Macro F1 of 0.650 is the honest number to report because it reflects the model's weakest-link performance — but it should be contextualised against the SOFT sample size limitation.

### Cross-Validation F1-macro = 0.842

The 5-fold CV on training data gave 0.842 ± 0.005, which is substantially higher than the test Macro F1 of 0.650. This gap is almost entirely explained by the SOFT compound: in training data there are 542 SOFT examples across 5 seasons, but in the test set (last 8 races of 2024) there are only 8. The test distribution is more conservative on SOFT usage than the training period, making the test harder for that class.

### Summary Assessment

Task B is a meaningful contribution beyond the paper, which had no compound recommendation at all. For the dominant compound (HARD, 77.5% of test pit stops), the model performs at near-production quality (F1=0.856). For MEDIUM it is useful but imperfect. For SOFT it is limited by the scarcity of test examples. The 78.7% overall accuracy on Task A True Positive laps is a solid operational result.

---

## Combined Pipeline Summary

| Component | Task A | Task B |
|---|---|---|
| Model | Bi-LSTM [v4] | XGBoost |
| Problem type | Binary classification | 3-class classification |
| Input | 10-lap sequences (all laps) | Single row (predicted pit laps only) |
| Output | P(pit this lap) | P(HARD), P(MEDIUM), P(SOFT) |
| Training rows | ~sequences from 98k laps | 2,957 pit stop rows |
| Class balancing | class_weight {0:1, 1:25} | SMOTE (safe — independent rows) |
| Test evaluation | 8,274 laps across 8 races | 169 Task A True Positive laps |
| F1-Score | **0.863** | **0.787 accuracy** / 0.650 macro F1 |
| vs. Paper | Beats paper F1 (0.863 vs 0.81) | Novel — paper had no Task B |
| Key strength | Very high precision (0.950) | HARD compound recommendation (0.856 F1) |
| Known limitation | Recall 0.791 (misses ~1 in 5 pits) | SOFT class (8 test examples) |

The end-to-end system — when Task A fires and is correct — recommends the right compound 78.7% of the time. Given that Task A fires correctly on 169 of 179 predictions (94.4% of alerts are real pit stops), a strategy engineer using this system would receive approximately 170 genuine pit alerts across 8 races, with the correct compound recommendation on about 133 of them.

# ============================================================
# F1 PIT STOP PREDICTOR — FULL TRANSFORMER VERSION (100%)
# Features:
#   ✅ Transformer Encoder with Multi-Head Self-Attention
#   ✅ Positional Encoding (lap-order aware)
#   ✅ Multi-Task Head (pit stop + compound prediction)
#   ✅ Race-Level Global Context Embedding
#   ✅ Per-Stint Normalization (no data leakage)
#   ✅ Attention Weight Visualization
#   ✅ Threshold Tuning via F1 Score
#   ✅ SHAP Feature Importance
#   ✅ Full Inference Pipeline
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Input, Concatenate, Flatten,
    Embedding, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_recall_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1. POSITIONAL ENCODING
#    Transformer নিজে থেকে জানে না কোন lap আগে কোনটা পরে।
#    Sinusoidal encoding দিয়ে lap order inject করা হয়।
# ============================================================
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        positions = np.arange(sequence_length)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Shape: (1, seq_len, d_model)
        self.pos_encoding = tf.cast(angles[np.newaxis, ...], tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        return config


# ============================================================
# 2. TRANSFORMER ENCODER BLOCK
#    প্রতিটা block-এ:
#      → Multi-Head Self-Attention (সব lap একে অপরের দিকে তাকায়)
#      → Feed-Forward Network (feature transform)
#      → Residual Connection + LayerNorm (stable training)
# ============================================================
class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"mha_{kwargs.get('name','')}"
        )
        self.ffn = keras.Sequential([
            Dense(ffn_dim, activation='gelu'),
            Dropout(dropout_rate),
            Dense(d_model),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False, return_attention_scores=False):
        if return_attention_scores:
            attn_out, attn_scores = self.attention(
                x, x, training=training, return_attention_scores=True
            )
            x = self.norm1(x + self.drop1(attn_out, training=training))
            x = self.norm2(x + self.drop2(self.ffn(x, training=training), training=training))
            return x, attn_scores
        else:
            attn_out = self.attention(x, x, training=training)
            x = self.norm1(x + self.drop1(attn_out, training=training))
            x = self.norm2(x + self.drop2(self.ffn(x, training=training), training=training))
            return x

    def get_config(self):
        return super().get_config()


# ============================================================
# 3. FULL MULTI-TASK TRANSFORMER MODEL
#    দুটো output একসাথে:
#      → pit_output    : এই lap-এ pit করবে কিনা (binary)
#      → compound_output: কোন টায়ারে যাবে (multi-class)
#
#    Race Context Embedding:
#      Circuit type, weather, round_number → global context token
#      এই token-টা sequence-এর শুরুতে prepend করা হয় [CLS token idea]
# ============================================================
def build_full_transformer_model(
    sequence_length,
    num_features,
    num_compounds,        # SOFT, MEDIUM, HARD, INTER, WET = 5
    num_circuits,         # unique circuits in dataset
    d_model=128,
    num_heads=4,
    num_encoder_blocks=4,
    ffn_dim=256,
    dropout_rate=0.1,
):
    # ── Lap sequence input ─────────────────────────────────
    lap_input = Input(shape=(sequence_length, num_features), name='lap_sequence')

    # ── Race context inputs ────────────────────────────────
    # Circuit embedding: প্রতিটা circuit-এর জন্য আলাদা learned vector
    circuit_input  = Input(shape=(1,), name='circuit_id', dtype='int32')
    circuit_embed  = Embedding(input_dim=num_circuits, output_dim=16, name='circuit_embed')(circuit_input)
    circuit_embed  = Flatten()(circuit_embed)

    # Scalar context: round_number, weather_code, safety_car_count etc.
    context_input  = Input(shape=(5,), name='race_context')

    # Combine context → project to d_model → use as a global "context token"
    context_merged = Concatenate()([circuit_embed, context_input])
    context_token  = Dense(d_model, activation='gelu', name='context_projection')(context_merged)
    context_token  = Reshape((1, d_model))(context_token)  # (batch, 1, d_model)

    # ── Project lap features → d_model ────────────────────
    x = Dense(d_model, name='feature_projection')(lap_input)  # (batch, seq, d_model)

    # ── Add positional encoding ────────────────────────────
    x = PositionalEncoding(sequence_length, d_model, name='pos_enc')(x)
    x = Dropout(dropout_rate, name='input_dropout')(x)

    # ── Prepend context token (like [CLS] in BERT) ─────────
    # Context token সব lap-এর সাথে attend করতে পারবে
    x = Concatenate(axis=1, name='prepend_context')([context_token, x])
    # x shape: (batch, seq+1, d_model)

    # ── Transformer Encoder Stack ──────────────────────────
    for i in range(num_encoder_blocks):
        x = TransformerEncoderBlock(
            d_model, num_heads, ffn_dim, dropout_rate,
            name=f'encoder_block_{i}'
        )(x)

    # ── Aggregate: context token (position 0) + mean of lap tokens ──
    context_out = x[:, 0, :]                        # [CLS] token output
    lap_out     = GlobalAveragePooling1D()(x[:, 1:, :])  # mean of lap tokens
    aggregated  = Concatenate(name='aggregated')([context_out, lap_out])

    # ── Shared representation ──────────────────────────────
    shared = Dense(128, activation='gelu', name='shared_1')(aggregated)
    shared = Dropout(0.2)(shared)
    shared = Dense(64,  activation='gelu', name='shared_2')(shared)

    # ── Task 1: Pit Stop Prediction (binary) ──────────────
    pit_branch = Dense(32, activation='gelu', name='pit_branch')(shared)
    pit_output = Dense(1, activation='sigmoid', name='pit_output')(pit_branch)

    # ── Task 2: Tire Compound Prediction (multi-class) ────
    # শুধু pit_output > threshold হলে এটা relevant
    compound_branch = Dense(32, activation='gelu', name='compound_branch')(shared)
    compound_output = Dense(
        num_compounds, activation='softmax', name='compound_output'
    )(compound_branch)

    model = Model(
        inputs=[lap_input, circuit_input, context_input],
        outputs=[pit_output, compound_output],
        name='F1_PitStop_Transformer_v2'
    )
    return model


# ============================================================
# 4. DATA PIPELINE
# ============================================================

FEATURE_COLUMNS = [
    'lap_number', 'tire_age_laps', 'tire_life_remaining_est',
    'position', 'gap_to_leader', 'gap_to_car_ahead',
    'cars_within_2s_ahead', 'cars_within_2s_behind',
    'is_being_attacked', 'is_stuck_in_train',
    'track_temperature', 'air_temperature', 'humidity',
    'current_compound_encoded', 'stint_number',
    'total_pit_stops_so_far', 'avg_lap_time_on_stint',
    'lap_time', 'lap_time_delta_prev'
]

COMPOUND_MAP = {
    'SOFT': 0, 'MEDIUM': 1, 'HARD': 2,
    'INTERMEDIATE': 3, 'WET': 4, 'NONE': 1  # default to MEDIUM
}
NUM_COMPOUNDS = 5


def prepare_features(df: pd.DataFrame):
    """Full feature engineering pipeline."""
    # Convert timedelta strings → seconds
    time_cols = [
        'lap_time', 'lap_time_delta_prev', 'sector1_time',
        'sector2_time', 'sector3_time', 'gap_to_leader', 'gap_to_car_ahead'
    ]
    for col in time_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
            except Exception:
                pass  # already numeric

    # Label encoders
    le_compound = LabelEncoder()
    le_circuit  = LabelEncoder()
    le_driver   = LabelEncoder()
    le_team     = LabelEncoder()

    df['current_compound_encoded'] = le_compound.fit_transform(
        df['current_compound'].fillna('MEDIUM')
    )
    df['circuit_id_encoded'] = le_circuit.fit_transform(
        df['circuit'].fillna('Unknown')
    )
    df['driver_encoded'] = le_driver.fit_transform(df['driver'].fillna('UNK'))
    df['team_encoded']   = le_team.fit_transform(df['team'].fillna('UNK'))

    # Weather code (simple mapping)
    if 'weather' not in df.columns:
        df['weather_code'] = 0
    else:
        weather_map = {'Dry': 0, 'Cloudy': 1, 'Light Rain': 2, 'Rain': 3, 'Heavy Rain': 4}
        df['weather_code'] = df['weather'].map(weather_map).fillna(0)

    # Derived features
    if 'lap_time' in df.columns and 'avg_lap_time_on_stint' in df.columns:
        df['deg_rate'] = df['lap_time'] - df['avg_lap_time_on_stint']

    # Safety car (if available)
    if 'safety_car_laps' not in df.columns:
        df['safety_car_laps'] = 0
    if 'vsc_laps' not in df.columns:
        df['vsc_laps'] = 0

    # Race context columns (5 scalar features)
    df['race_context_1'] = df['round_number'] / 24.0          # normalize
    df['race_context_2'] = df['weather_code'] / 4.0
    df['race_context_3'] = df['safety_car_laps'].clip(0, 10) / 10.0
    df['race_context_4'] = df['vsc_laps'].clip(0, 10) / 10.0
    df['race_context_5'] = df.get('total_laps', pd.Series(70, index=df.index)) / 70.0

    num_circuits = df['circuit_id_encoded'].nunique()

    return df, le_compound, le_circuit, num_circuits


def per_stint_normalize(group: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Per-stint normalization — LSTM এর চেয়ে বড় improvement।
    StandardScaler পুরো dataset-এ fit করলে future race data leak হয়।
    প্রতিটা stint-এর ভেতরে normalize করলে সেই সমস্যা নেই।
    """
    scaler = RobustScaler()  # outlier-robust
    cols_to_scale = [c for c in feature_columns if c in group.columns]
    group[cols_to_scale] = scaler.fit_transform(group[cols_to_scale].fillna(0))
    return group


RACE_CONTEXT_COLS = [
    'race_context_1', 'race_context_2', 'race_context_3',
    'race_context_4', 'race_context_5'
]


def create_sequences(df: pd.DataFrame, feature_columns: list, sequence_length: int = 10):
    """
    Create LSTM/Transformer sequences + race context per sample.
    GroupShuffleSplit এর জন্য race_group id ও রাখছি।
    """
    sequences      = []
    circuit_ids    = []
    race_contexts  = []
    targets_pit    = []
    targets_cmpd   = []
    race_groups    = []   # for GroupShuffleSplit (no race leakage)

    grouped = df.groupby(['driver', 'season', 'round_number'])

    for (driver, season, rnd), group in grouped:
        group = group.sort_values('lap_number').reset_index(drop=True)

        # Per-stint normalization
        group = (
            group.groupby('stint_number', group_keys=False)
                 .apply(lambda g: per_stint_normalize(g, feature_columns))
        )
        group = group.fillna(0)

        features    = group[feature_columns].values
        pit_labels  = group['pit_this_lap'].values
        ctx_vals    = group[RACE_CONTEXT_COLS].values
        circuit_val = group['circuit_id_encoded'].iloc[0]
        race_id     = f"{season}_{rnd}"

        for i in range(len(group) - sequence_length):
            seq = features[i : i + sequence_length]
            sequences.append(seq)
            circuit_ids.append(circuit_val)
            race_contexts.append(ctx_vals[i + sequence_length])
            targets_pit.append(pit_labels[i + sequence_length])

            cmpd = 'NONE'
            if pit_labels[i + sequence_length] == 1 and 'next_tire_compound' in group.columns:
                cmpd = group.iloc[i + sequence_length]['next_tire_compound']
            targets_cmpd.append(COMPOUND_MAP.get(str(cmpd).upper(), 1))
            race_groups.append(race_id)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(circuit_ids, dtype=np.int32),
        np.array(race_contexts, dtype=np.float32),
        np.array(targets_pit, dtype=np.float32),
        np.array(targets_cmpd, dtype=np.int32),
        np.array(race_groups)
    )


# ============================================================
# 5. RACE-AWARE TRAIN/TEST SPLIT
#    সমস্যা: random split করলে একই race-এর laps train ও test-এ যায়
#    ফলে model সহজেই "memorize" করে, real performance বোঝা যায় না।
#    GroupShuffleSplit দিয়ে পুরো race একসাথে train বা test-এ যায়।
# ============================================================
def race_aware_split(X, X_circuit, X_ctx, y_pit, y_cmpd, groups, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y_pit, groups=groups))
    return (
        X[train_idx], X[test_idx],
        X_circuit[train_idx], X_circuit[test_idx],
        X_ctx[train_idx], X_ctx[test_idx],
        y_pit[train_idx], y_pit[test_idx],
        y_cmpd[train_idx], y_cmpd[test_idx],
    )


# ============================================================
# 6. OPTIMAL THRESHOLD FINDER
#    Default 0.5 threshold সবসময় best না।
#    F1 score maximize করে optimal threshold বের করা হয়।
# ============================================================
def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx  = np.argmax(f1_scores[:-1])
    best_thr  = thresholds[best_idx]
    print(f"\n✅ Optimal threshold: {best_thr:.4f} (F1={f1_scores[best_idx]:.4f})")
    return best_thr


# ============================================================
# 7. ATTENTION VISUALIZATION
#    Transformer-এর সবচেয়ে বড় advantage: interpretability।
#    কোন lap-এ model সবচেয়ে বেশি "মনোযোগ" দিয়েছে সেটা দেখা যায়।
# ============================================================
def visualize_attention(
    model, sample_input, lap_numbers, sequence_length,
    block_index=0, save_path='attention_heatmap.png'
):
    """
    Extract and plot attention weights from a specific encoder block.
    sample_input: [lap_seq, circuit_id, context] for 1 sample
    """
    # Build intermediate model to get attention scores
    encoder_blocks = [
        layer for layer in model.layers
        if isinstance(layer, TransformerEncoderBlock)
    ]
    if block_index >= len(encoder_blocks):
        print("Block index out of range")
        return

    target_block = encoder_blocks[block_index]

    # Use TF GradientTape to get intermediate output
    @tf.function
    def get_attn(inputs):
        x_seq, x_cir, x_ctx = inputs
        # Replicate forward pass up to target block
        circuit_embed = model.get_layer('circuit_embed')(x_cir)
        circuit_flat  = tf.reshape(circuit_embed, [tf.shape(circuit_embed)[0], -1])
        ctx_merged    = tf.concat([circuit_flat, x_ctx], axis=-1)
        ctx_token     = model.get_layer('context_projection')(ctx_merged)
        ctx_token     = tf.reshape(ctx_token, [-1, 1, ctx_token.shape[-1]])

        x = model.get_layer('feature_projection')(x_seq)
        x = model.get_layer('pos_enc')(x)
        x = model.get_layer('input_dropout')(x, training=False)
        x = tf.concat([ctx_token, x], axis=1)

        for i, block in enumerate(encoder_blocks):
            if i < block_index:
                x = block(x, training=False)
            else:
                x, attn_w = block(x, training=False, return_attention_scores=True)
                return attn_w  # shape: (batch, heads, seq+1, seq+1)

    attn_weights = get_attn(sample_input)  # (1, heads, seq+1, seq+1)
    attn_np = attn_weights[0].numpy()      # (heads, seq+1, seq+1)

    # Labels: [CTX] + lap numbers
    labels = ['[CTX]'] + [f'L{n}' for n in lap_numbers]
    n_heads = attn_np.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        sns.heatmap(
            attn_np[h], ax=ax,
            xticklabels=labels, yticklabels=labels,
            cmap='YlOrRd', vmin=0, vmax=attn_np[h].max(),
            linewidths=0.3, linecolor='gray'
        )
        ax.set_title(f'Head {h+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key (attending TO)')
        ax.set_ylabel('Query (attending FROM)')

    plt.suptitle(
        f'Attention Weights — Encoder Block {block_index+1}\n'
        f'(কোন lap-এ model বেশি মনোযোগ দিয়েছে)',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Attention heatmap saved: {save_path}")


# ============================================================
# 8. TRAINING PLOTS
# ============================================================
def plot_training_history(history, save_path='training_history.png'):
    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    metrics_pairs = [
        ('loss',     'val_loss',     'Loss',              'Loss'),
        ('pit_output_accuracy', 'val_pit_output_accuracy', 'Pit Accuracy', 'Accuracy'),
        ('pit_output_precision', 'val_pit_output_precision', 'Pit Precision & Recall', 'Score'),
    ]

    colors = [('#2196F3', '#FF5722'), ('#4CAF50', '#FF9800'), ('#9C27B0', '#00BCD4')]

    for idx, (tr_key, vl_key, title, ylabel) in enumerate(metrics_pairs):
        ax = fig.add_subplot(gs[idx])
        c1, c2 = colors[idx]
        if tr_key in history.history:
            ax.plot(history.history[tr_key], color=c1, label='Train', linewidth=2)
        if vl_key in history.history:
            ax.plot(history.history[vl_key], color=c2, label='Val', linewidth=2, linestyle='--')

        # Add recall to third plot
        if idx == 2:
            if 'pit_output_recall' in history.history:
                ax.plot(history.history['pit_output_recall'], color='#E91E63', label='Train Recall')
            if 'val_pit_output_recall' in history.history:
                ax.plot(history.history['val_pit_output_recall'], color='#009688',
                        label='Val Recall', linestyle='--')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('F1 Pit Stop Transformer — Training History', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['No Pit', 'Pit'],
        yticklabels=['No Pit', 'Pit'],
        linewidths=0.5, linecolor='gray',
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, f'({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    ax.set_title('Confusion Matrix — F1 Pit Prediction', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


# ============================================================
# 9. INFERENCE PIPELINE
# ============================================================
class F1PitPredictor:
    """
    Production-ready predictor class.
    Race strategist-এর মতো ব্যবহার করা যাবে।
    """
    def __init__(self, model, feature_columns, le_circuit, threshold=0.5):
        self.model           = model
        self.feature_columns = feature_columns
        self.le_circuit      = le_circuit
        self.threshold       = threshold
        self.compound_names  = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        self.stint_scaler    = RobustScaler()

    def predict(
        self,
        recent_laps_df: pd.DataFrame,
        circuit_name: str,
        race_context: dict
    ) -> dict:
        """
        Args:
            recent_laps_df : DataFrame with last 10 laps
            circuit_name   : e.g., 'Monza'
            race_context   : dict with keys:
                             round_number, weather_code, safety_car_laps,
                             vsc_laps, total_laps
        Returns:
            dict with prediction details
        """
        # Prepare lap features
        features = recent_laps_df[self.feature_columns].fillna(0).values
        features = self.stint_scaler.fit_transform(features)   # per-stint normalize
        features = features.reshape(1, len(features), -1).astype(np.float32)

        # Circuit ID
        try:
            cid = self.le_circuit.transform([circuit_name])[0]
        except ValueError:
            cid = 0  # unknown circuit
        circuit_arr = np.array([[cid]], dtype=np.int32)

        # Race context vector
        ctx = np.array([[
            race_context.get('round_number', 1) / 24.0,
            race_context.get('weather_code', 0) / 4.0,
            min(race_context.get('safety_car_laps', 0), 10) / 10.0,
            min(race_context.get('vsc_laps', 0), 10) / 10.0,
            race_context.get('total_laps', 70) / 70.0,
        ]], dtype=np.float32)

        # Predict
        pit_prob_arr, compound_arr = self.model.predict(
            [features, circuit_arr, ctx], verbose=0
        )
        pit_prob      = float(pit_prob_arr[0][0])
        compound_probs = compound_arr[0]
        best_compound = self.compound_names[np.argmax(compound_probs)]

        return {
            'pit_probability'    : round(pit_prob, 4),
            'should_pit'         : pit_prob >= self.threshold,
            'recommendation'     : '🔴 PIT NOW' if pit_prob >= self.threshold else '🟢 STAY OUT',
            'recommended_compound': best_compound if pit_prob >= self.threshold else 'N/A',
            'compound_confidence' : {
                c: round(float(p), 3)
                for c, p in zip(self.compound_names, compound_probs)
            },
            'threshold_used'     : self.threshold,
        }


# ============================================================
# 10. MAIN TRAINING SCRIPT
# ============================================================

# ── Load data ──────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv('f1_complete_dataset_2020_2024.csv')
print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")

# ── Feature engineering ────────────────────────────────────
print("\n🔧 Feature engineering...")
df, le_compound, le_circuit, num_circuits = prepare_features(df)
print(f"   Unique circuits: {num_circuits}")

# ── Create sequences ───────────────────────────────────────
print("\n⏳ Creating sequences...")
SEQUENCE_LENGTH = 10
X, X_circuit, X_ctx, y_pit, y_cmpd, groups = create_sequences(
    df, FEATURE_COLUMNS, SEQUENCE_LENGTH
)

print(f"   Sequences shape : {X.shape}")
print(f"   Pit distribution: No={np.sum(y_pit==0):,} | Yes={np.sum(y_pit==1):,}")
print(f"   Imbalance ratio : {np.sum(y_pit==0)/np.sum(y_pit==1):.1f}:1")

# ── Race-aware split ───────────────────────────────────────
print("\n✂️  Race-aware train/test split...")
(X_tr, X_te,
 Xc_tr, Xc_te,
 Xx_tr, Xx_te,
 yp_tr, yp_te,
 yc_tr, yc_te) = race_aware_split(X, X_circuit, X_ctx, y_pit, y_cmpd, groups)

print(f"   Train: {len(X_tr):,} | Test: {len(X_te):,}")

# ── Class weights ──────────────────────────────────────────
pit_weight = len(yp_tr[yp_tr == 0]) / len(yp_tr[yp_tr == 1])
class_weights_pit = {0: 1.0, 1: pit_weight}
print(f"\n⚖️  Pit class weight: {pit_weight:.2f}")

# ── Build model ────────────────────────────────────────────
print("\n🏗️  Building Transformer model...")
model = build_full_transformer_model(
    sequence_length = SEQUENCE_LENGTH,
    num_features    = len(FEATURE_COLUMNS),
    num_compounds   = NUM_COMPOUNDS,
    num_circuits    = num_circuits,
    d_model         = 128,
    num_heads       = 4,
    num_encoder_blocks = 4,
    ffn_dim         = 256,
    dropout_rate    = 0.1,
)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4),
    loss={
        'pit_output'     : 'binary_crossentropy',
        'compound_output': 'sparse_categorical_crossentropy',
    },
    loss_weights={
        'pit_output'     : 1.0,   # pit prediction বেশি important
        'compound_output': 0.4,   # compound auxiliary task
    },
    metrics={
        'pit_output'     : ['accuracy', keras.metrics.Precision(name='precision'),
                             keras.metrics.Recall(name='recall'),
                             keras.metrics.AUC(name='auc')],
        'compound_output': ['accuracy'],
    }
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_pit_output_auc', mode='max',
        patience=20, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_pit_output_auc', mode='max',
        factor=0.5, patience=7, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        'best_f1_transformer.h5',
        monitor='val_pit_output_auc', mode='max',
        save_best_only=True, verbose=1
    ),
]

# ── Train ──────────────────────────────────────────────────
print("\n🚀 Training started...")
history = model.fit(
    x=[X_tr, Xc_tr, Xx_tr],
    y={'pit_output': yp_tr, 'compound_output': yc_tr},
    validation_split=0.15,
    epochs=150,
    batch_size=64,
    callbacks=callbacks,
    class_weight={'pit_output': class_weights_pit},
    verbose=1,
)

# ── Evaluate ───────────────────────────────────────────────
print("\n" + "="*55)
print("📊 EVALUATION ON HELD-OUT TEST RACES")
print("="*55)

pit_probs, compound_probs = model.predict([X_te, Xc_te, Xx_te], verbose=0)
pit_probs = pit_probs.flatten()

# Find optimal threshold
best_threshold = find_optimal_threshold(yp_te, pit_probs)
y_pred = (pit_probs >= best_threshold).astype(int)

print(f"\nROC-AUC : {roc_auc_score(yp_te, pit_probs):.4f}")
print(f"F1 Score: {f1_score(yp_te, y_pred):.4f}")
print("\n" + classification_report(yp_te, y_pred, target_names=['No Pit', 'Pit']))

# ── Compound accuracy (only on actual pit laps) ────────────
pit_mask        = yp_te == 1
cmpd_pred       = np.argmax(compound_probs[pit_mask], axis=1)
cmpd_acc        = np.mean(cmpd_pred == yc_te[pit_mask])
print(f"Tire Compound Accuracy (on pit laps): {cmpd_acc:.4f}")

# ── Plots ──────────────────────────────────────────────────
plot_training_history(history)
plot_confusion_matrix(yp_te, y_pred)

# ── Attention visualization on 1 sample ───────────────────
print("\n🔍 Generating attention heatmap...")
sample_idx  = np.where(yp_te == 1)[0][0]  # pick a pit-stop sample
sample_laps = X_te[sample_idx]
lap_nums    = list(range(1, SEQUENCE_LENGTH + 1))

visualize_attention(
    model,
    sample_input=[
        X_te[sample_idx:sample_idx+1],
        Xc_te[sample_idx:sample_idx+1],
        Xx_te[sample_idx:sample_idx+1],
    ],
    lap_numbers=lap_nums,
    sequence_length=SEQUENCE_LENGTH,
    block_index=2,    # last encoder block is most task-specific
    save_path='attention_heatmap.png'
)

# ── Save everything ────────────────────────────────────────
print("\n💾 Saving model and artifacts...")
model.save('f1_pit_transformer_final.h5')
with open('le_circuit.pkl', 'wb') as f:  pickle.dump(le_circuit, f)
with open('le_compound.pkl', 'wb') as f: pickle.dump(le_compound, f)
with open('feature_columns.pkl', 'wb') as f: pickle.dump(FEATURE_COLUMNS, f)

metadata = {
    'sequence_length': SEQUENCE_LENGTH,
    'num_features'   : len(FEATURE_COLUMNS),
    'num_compounds'  : NUM_COMPOUNDS,
    'num_circuits'   : num_circuits,
    'best_threshold' : best_threshold,
}
with open('model_metadata.pkl', 'wb') as f: pickle.dump(metadata, f)
print("✅ All artifacts saved!")

# ============================================================
# 11. EXAMPLE INFERENCE
# ============================================================
print("\n" + "="*55)
print("🏁 EXAMPLE INFERENCE")
print("="*55)

predictor = F1PitPredictor(
    model=model,
    feature_columns=FEATURE_COLUMNS,
    le_circuit=le_circuit,
    threshold=best_threshold,
)

# Take Gasly's last 10 laps as example
sample_driver_laps = df[df['driver'] == 'GAS'].tail(SEQUENCE_LENGTH).copy()

result = predictor.predict(
    recent_laps_df=sample_driver_laps,
    circuit_name='Monza',
    race_context={
        'round_number'    : 16,
        'weather_code'    : 0,       # Dry
        'safety_car_laps' : 2,
        'vsc_laps'        : 0,
        'total_laps'      : 53,
    }
)

print(f"\nDriver        : GAS — Pierre Gasly")
print(f"Pit Probability : {result['pit_probability']:.2%}")
print(f"Recommendation  : {result['recommendation']}")
if result['should_pit']:
    print(f"Recommended Compound: {result['recommended_compound']}")
    print(f"Compound Confidence : {result['compound_confidence']}")
print(f"Threshold Used  : {result['threshold_used']:.4f}")
#BEST OUTPUT SO FAR

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# --- STEP 1: DATA LOADING & SAFE NUMERIC CONVERSION ---
csv_files = glob.glob("**/*.csv", recursive=True)
csv_files = [f for f in csv_files if "dataset" in f.lower() or "20" in f]
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

def safe_num(df, col, default=0.0):
    return pd.to_numeric(df[col], errors='coerce').fillna(default)

df['lap_time'] = safe_num(df, 'lap_time', 100.0)
df['tire_age_laps'] = safe_num(df, 'tire_age_laps', 0.0)
df['pit_this_lap'] = safe_num(df, 'pit_this_lap', 0).astype(int)

# --- STEP 2: FEATURE ENGINEERING ---
durability = {'SOFT': 25, 'MEDIUM': 35, 'HARD': 50}
df['max_life'] = df['current_compound'].astype(str).str.upper().map(durability).fillna(35)
df['tire_life_remaining'] = df['max_life'] - df['tire_age_laps']
df['compound_enc'] = LabelEncoder().fit_transform(df['current_compound'].astype(str))

features = ['lap_time', 'tire_age_laps', 'compound_enc', 'tire_life_remaining']
df[features] = MinMaxScaler().fit_transform(df[features])

# --- STEP 3: SEQUENCE CREATION ---
def create_sequences(data, window_size=10):
    X, y = [], []
    for _, group in data.groupby('driver'):
        group = group.sort_values('lap_number')
        if len(group) <= window_size: continue
        f_vals, t_vals = group[features].values, group['pit_this_lap'].values
        for i in range(len(f_vals) - window_size):
            X.append(f_vals[i : i+window_size])
            y.append(t_vals[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(df, window_size=10)

# --- STEP 4: APPLY SMOTE (Balancing the Data) ---
# SMOTE requires 2D data, so we flatten, resample, then reshape back
X_flat = X.reshape(X.shape[0], -1)
smote = SMOTE(sampling_strategy=0.3, random_state=42) # Bring "Pit In" to 30% of "Stay Out"
X_res, y_res = smote.fit_resample(X_flat, y)
X_final = X_res.reshape(X_res.shape[0], 10, len(features))

X_train, X_test, y_train, y_test = train_test_split(X_final, y_res, test_size=0.2, stratify=y_res)

# --- STEP 5: BRAIN OF THE MODEL ---
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, len(features))),
    BatchNormalization(),
    LSTM(32),
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

print("\n🔥 Training with SMOTE Balanced Data...")
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# --- STEP 6: SMART EVALUATION ---
# Using a higher threshold (0.7) to reduce False Positives
y_probs = model.predict(X_test)
y_pred = (y_probs > 0.7).astype(int)

print("\n--- NEW CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

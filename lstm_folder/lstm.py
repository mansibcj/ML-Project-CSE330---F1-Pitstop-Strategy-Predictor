
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- STEP 1: AUTO-DETECT FILES (RECURSIVE SEARCH) ---
# We use recursive=True to look inside folders like 'DATA_SET_BUILDER'
print("--- Searching for CSV files in all folders ---")
csv_files = glob.glob("**/*.csv", recursive=True)

# Filter out system files or irrelevant csvs if any (optional)
csv_files = [f for f in csv_files if "dataset" in f.lower() or "20" in f]

if not csv_files:
    print("\n Error: Still no CSV files found.")
    print("Files found in current directory:", os.listdir())
    print("If you are in Colab, try dragging the files out of the folder and into the main area.")
    # Fallback: Force upload if search fails
    try:
        from google.colab import files
        print("\n Launching Manual Upload as fallback... ")
        uploaded = files.upload()
        csv_files = list(uploaded.keys())
    except:
        raise ValueError("Could not find files automatically and manual upload failed.")
else:
    print(f"\n Found {len(csv_files)} datasets: {csv_files}")

# --- STEP 2: LOAD DATA ---
data_frames = []
for f in csv_files:
    try:
        df_temp = pd.read_csv(f)
        data_frames.append(df_temp)
        print(f"-> Loaded {f}: {len(df_temp)} rows")
    except Exception as e:
        print(f"Skipping {f}: {e}")

if not data_frames:
    raise ValueError("No valid data loaded.")

full_data = pd.concat(data_frames, ignore_index=True)

# --- STEP 3: PREPARE DATA ---
# Use the columns from your specific dataset
cols_needed = ['driver', 'lap_number', 'lap_time', 'tire_age_laps', 'current_compound', 'track_status', 'pit_this_lap']

# Check if columns exist
missing = [c for c in cols_needed if c not in full_data.columns]
if missing:
    print(f" Warning: Your CSV is missing these columns: {missing}")
    print("Available columns:", full_data.columns.tolist())
    raise ValueError("Please check your dataset column names.")

df = full_data[cols_needed].copy()
df.dropna(inplace=True)
df = df[df['lap_time'] < 200] # Filter safety cars/red flags

# Encode Compounds
le = LabelEncoder()
df['current_compound'] = df['current_compound'].astype(str)
df['compound_encoded'] = le.fit_transform(df['current_compound'])

# Scale Features
scaler = MinMaxScaler()
features = ['lap_time', 'tire_age_laps', 'compound_encoded', 'track_status']
df[features] = scaler.fit_transform(df[features])

print(f"Training Data Ready: {len(df)} laps")

# --- STEP 4: CREATE LSTM SEQUENCES ---
def create_sequences(data, window_size=5):
    X, y = [], []
    # Group by Driver so we don't mix up different cars
    grouped = data.groupby('driver')
    
    for _, group in grouped:
        group = group.sort_values('lap_number')
        vals = group[features].values
        targets = group['pit_this_lap'].values
        
        if len(group) < window_size + 1:
            continue
            
        for i in range(len(vals) - window_size - 1):
            X.append(vals[i : i+window_size])
            y.append(targets[i+window_size+1]) # Predict NEXT lap
            
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = create_sequences(df, WINDOW_SIZE)

# --- STEP 5: TRAIN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Calculate weights to handle rare pit stops
weights = class_weight.compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
cw = dict(enumerate(weights))

print("\nStarting Training...")
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    class_weight=cw,
    verbose=1
)

# --- STEP 6: RESULTS ---
print("\n--- METRICS ---")
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['Stay Out', 'Pit In']))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_samples = 8000

# Synthetic feature generation reflecting ground reality
# segment_idx: 0 (NDLS-GZB), 1 (GZB-ALJN), 2 (ALJN-CNB)
segment_idx = np.random.choice([0, 1, 2], size=n_samples)
nominal_times = np.array([18.0, 55.0, 160.0])[segment_idx]
distances = np.array([25.0, 105.0, 305.0])[segment_idx]

# Congestion features
headway_km = np.random.uniform(1.0, 20.0, size=n_samples)
# 0: Red, 1: Yellow, 2: Double Yellow, 3: Green
signal_aspect = np.random.choice([0, 1, 2, 3], p=[0.08, 0.17, 0.25, 0.50], size=n_samples)
active_tsr_kmph = np.random.choice([0, 30, 50, 75], p=[0.65, 0.15, 0.10, 0.10], size=n_samples)
block_density = np.random.randint(1, 6, size=n_samples) # trains in sector

# Target calculation: Actual traversal duration with non-linear rail physics
delays = np.zeros(n_samples)
delays += (signal_aspect == 0) * np.random.uniform(12.0, 25.0, size=n_samples)
delays += (signal_aspect == 1) * np.random.uniform(5.0, 12.0, size=n_samples)
delays += (headway_km < 3.0) * (4.0 / (headway_km + 0.1))
delays += (active_tsr_kmph > 0) * ((distances / np.maximum(active_tsr_kmph, 1)) * 60 - nominal_times) * 0.4
delays += block_density * np.random.uniform(1.0, 3.0, size=n_samples)
delays = np.maximum(delays, 0.0)

actual_duration_min = nominal_times + delays

X = pd.DataFrame({
    'segment_idx': segment_idx,
    'nominal_time_min': nominal_times,
    'headway_km': headway_km,
    'signal_aspect': signal_aspect,
    'active_tsr_kmph': active_tsr_kmph,
    'block_density': block_density
})
y = actual_duration_min

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = LGBMRegressor(n_estimators=120, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Model successfully trained and saved. Test R2 score: {model.score(X_test, y_test):.4f}")
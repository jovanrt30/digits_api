from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Buat folder model jika belum ada
os.makedirs("model", exist_ok=True)

# 1. Load data
digits = load_digits()
X, y = digits.data, digits.target

# 2. Split data train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Latih model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluasi model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy:.4f}")

# 5. Simpan model
joblib.dump(clf, "model/clf_digits.pkl")
print("Model disimpan di model/clf_digits.pkl")

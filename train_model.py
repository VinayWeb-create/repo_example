import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    data_path = "data/keystroke_features.csv"

    if not os.path.exists(data_path):
        print("❌ Feature CSV not found. Run extract_features.py first.")
        return

    df = pd.read_csv(data_path)

    if len(df['username'].unique()) < 2:
        print("⚠ You need at least 2 different users to train the model properly.")
        return

    # Encode username as numeric labels
    le = LabelEncoder()
    df['user_id'] = le.fit_transform(df['username'])

    X = df[['avg_hold_time', 'avg_flight_time', 'total_keys', 'total_time']]
    y = df['user_id']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model & label encoder
    joblib.dump(model, 'model/keystroke_model.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')
    print("✅ Model & label encoder saved in /model")

if __name__== "__main__":
    os.makedirs('model', exist_ok=True)
    train_model()
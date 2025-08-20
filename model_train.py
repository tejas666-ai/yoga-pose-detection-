import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv(r"C:/Users/Admin/OneDrive/Desktop/yoga project new/combined_yoga_pose_angles.csv")

# Drop 'Unnamed: 0' and any image/file path columns
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Automatically drop any non-numeric columns except the label
label_col = "label"  # Change if your label column has a different name

# Ensure label column is at the end
cols = list(df.columns)
if label_col in cols:
    cols.remove(label_col)
    cols.append(label_col)
    df = df[cols]

# Separate features and label
X = df.drop(columns=[label_col])
y = df[label_col]

# Ensure only numeric data in X
X = X.select_dtypes(include=["number"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model,r"C:/Users/Admin/OneDrive/Desktop/yoga project new/yoga_pose_classifier.joblib")
print("✅ Model saved as 'yoga_pose_classifier.joblib'")

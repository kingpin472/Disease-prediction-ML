# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load the dataset
df = pd.read_csv("diabetes.csv")  # Make sure 'diabetes.csv' is in your working directory
print(df.head())

# Step 3: Basic data check
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Step 4: Data visualization
sns.countplot(x='Outcome', data=df)
plt.title('Diabetes Outcome Distribution')
plt.show()

# Step 5: Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 6: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 10: Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=df.columns[:-1])
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()

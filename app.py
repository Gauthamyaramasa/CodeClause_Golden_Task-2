import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv('training_dataset.csv')

# Feature selection: Use 'Age', 'Gender', and personality trait scores as features
X = df[['Age', 'Gender', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
y = df['Personality (Class label)']

# Encode non-numeric features like 'Gender'
label_encoders = {}
for col in ['Gender']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Random Forest in this example)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(classifier, 'personality_prediction_model.joblib')

# User input and prediction
def predict_personality(age, gender, openness, neuroticism, conscientiousness, agreeableness, extraversion):
    # Load the trained model
    classifier = joblib.load('personality_prediction_model.joblib')
    
    # Encode 'Gender' using the label encoder
    gender_encoded = label_encoders['Gender'].transform([gender])[0]
    
    # Create a DataFrame for user input features
    input_features = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'openness': [openness],
        'neuroticism': [neuroticism],
        'conscientiousness': [conscientiousness],
        'agreeableness': [agreeableness],
        'extraversion': [extraversion]
    })
    
    # Make a prediction
    predicted_personality = classifier.predict(input_features)
    return predicted_personality[0]

if __name__ == "__main__":
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (e.g., Male/Female): ")
    openness = int(input("Enter your openness score (0-10): "))
    neuroticism = int(input("Enter your neuroticism score (0-10): "))
    conscientiousness = int(input("Enter your conscientiousness score (0-10): "))
    agreeableness = int(input("Enter your agreeableness score (0-10): "))
    extraversion = int(input("Enter your extraversion score (0-10): "))
    
    predicted_personality = predict_personality(age, gender, openness, neuroticism, conscientiousness, agreeableness, extraversion)
    print(f"Predicted Personality: {predicted_personality}")

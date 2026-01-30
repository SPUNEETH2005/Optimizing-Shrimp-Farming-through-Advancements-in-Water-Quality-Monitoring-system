from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)



# Load and preprocess data, train model
def load_and_train_model(filename):
    # Try reading the file with different encodings
    encodings = ['utf-8', 'windows-1252']  # Added 'utf-8' as a common encoding
    data = None

    for encoding in encodings:
        try:
            data = pd.read_csv(filename, encoding=encoding)
            print(f"File successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to read with encoding: {encoding}")
            continue

    if data is None:
        raise ValueError("Unable to read the file with any of the specified encodings.")

    # Print column names to verify
    print("Columns in the CSV file:", data.columns.tolist())

    # Preprocess data
    data['Nutrient Level'] = data['Nutrient Level'].map({'Poor': 0, 'Good': 1, 'Excellent': 2})

    # Update feature columns to match the CSV file
    X = data[['Temperature (°C)', 'pH', 'OD (mg/L)', 'TDS (ppt)', 'Salinity (ppt)']]
    y = data['Nutrient Level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load the model
model = load_and_train_model('Shrimp.csv')

# Define limits and recommendations
limits = {
    'Temperature (°C)': (20, 35),
    'pH': (6.5, 8.5),
    'OD (mg/L)': (4, 8),
    'TDS (ppt)': (500, 3000),
    'Salinity (ppt)': (10, 35)
}

low_recommendations = {
    'Temperature (°C)': ["Increase the temperature to improve nutrient levels."],
    'pH': ["Add lime to increase pH levels."],
    'OD (mg/L)': ["Increase aeration to improve dissolved oxygen levels."],
    'TDS (ppt)': ["Add minerals to increase TDS levels."],
    'Salinity (ppt)': ["Add salt to increase salinity levels."]
}

high_recommendations = {
    'Temperature (°C)': ["Decrease the temperature to improve nutrient levels."],
    'pH': ["Add acid to decrease pH levels."],
    'OD (mg/L)': ["Reduce aeration to lower dissolved oxygen levels."],
    'TDS (ppt)': ["Dilute water to reduce TDS levels."],
    'Salinity (ppt)': ["Dilute water to reduce salinity levels."]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    recommendations = []

    # Processing form data
    try:
        for feature in limits.keys():
            value = float(request.form.get(feature))
            input_data[feature] = value

            # Check limits and add recommendations
            low, high = limits[feature]
            if value < low:
                recommendations.extend(low_recommendations[feature])
            elif value > high:
                recommendations.extend(high_recommendations[feature])
    except (ValueError, TypeError) as e:
        return render_template('index.html', result={'error': 'Invalid input. Please enter numeric values.'})

    # Predict using the trained model
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    nutrient_levels = {0: 'Poor', 1: 'Good', 2: 'Excellent'}
    predicted_level = nutrient_levels[prediction[0]]

    # Prepare result
    result = {
        'prediction': predicted_level,
        'recommendations': recommendations
    }


    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import traceback
from contextlib import asynccontextmanager
from typing import List

app = FastAPI()

# Global variables for the model and data
rf_classifier = None
data = None
X_encoded = None

# Define the data model for the user input
class UserInput(BaseModel):
    age: int
    height: float
    weight: float
    gender: str
    injury: str
    level: str
    equipment: str

# Define the data model for the exercise recommendation
class ExerciseRecommendation(BaseModel):
    exercises: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rf_classifier, data, X_encoded
    try:
        # Load your data
        data = pd.read_csv('Exercises.csv')
        data['gender_encoded'] = data['Gender'].map({'male': 0, 'female': 1})

        # Prepare the features and target variable
        X = data[['Age', 'Height', 'Weight', 'injury', 'gender_encoded']]
        y = data['Exercise']

        # Encode categorical features
        X_encoded = pd.get_dummies(X, columns=['injury'])

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Train the RandomForestClassifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        print("Model and data loaded successfully")
        
        yield
    except Exception as e:
        print(f"Error loading model or data: {e}")
        traceback.print_exc()
        yield
    finally:
        # You can perform any necessary cleanup here
        print("Cleaning up resources...")

app.router.lifespan_context = lifespan

# Function to predict exercise
def predict_exercise(model, user_input, X_encoded):
    try:
        # Prepare the user input for prediction
        user_input['gender_encoded'] = 1 if user_input['gender'].lower() == 'female' else 0
        user_input_df = pd.DataFrame([user_input])
        user_input_encoded = pd.get_dummies(user_input_df, columns=['injury'])
        user_input_encoded = user_input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
        
        # Predict exercise
        predicted_exercise = model.predict(user_input_encoded)[0]
        return predicted_exercise
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        raise

# Main function to get recommendation
def get_exercise_recommendation(user_input, model, data, X_encoded, n=5):
    try:
        predicted_exercise = predict_exercise(model, user_input, X_encoded)
        
        # Check if the predicted exercise matches the user's specifications
        predicted_exercise_row = data[data['Exercise'] == predicted_exercise]
        
        if not predicted_exercise_row.empty:
            predicted_exercise_matches = (
                (predicted_exercise_row['equipment'].values[0].lower() in ['dumbbells', 'body only']) if user_input['equipment'].lower() == 'dumbbells' else (predicted_exercise_row['equipment'].values[0].lower() == user_input['equipment'].lower()) and
                (predicted_exercise_row['level'].values[0].lower() in ['beginner', 'intermediate', 'advanced']) if user_input['level'].lower() == 'advanced' else (predicted_exercise_row['level'].values[0].lower() in ['beginner', 'intermediate']) if user_input['level'].lower() == 'intermediate' else (predicted_exercise_row['level'].values[0].lower() == user_input['level'].lower())
            )
            
            if predicted_exercise_matches:
                initial_recommendations = [predicted_exercise]
            else:
                initial_recommendations = []
        else:
            initial_recommendations = []
        
        # Filter exercises based on user's injury and level
        filtered_exercises = data[(data['injury'].str.lower() == user_input['injury'].lower()) & (data['level'].str.lower() == user_input['level'].lower())]
        
        # Filter exercises further based on user's equipment (if specified as 'body only')
        if user_input['equipment'].lower() == 'body only':
            filtered_exercises = filtered_exercises[filtered_exercises['equipment'].str.lower() == 'body only']
        
        # Get unique exercises from the filtered list
        unique_exercises = list(filtered_exercises['Exercise'].unique())
        
        # Initialize recommendations with predicted exercise if it matches the criteria
        recommendations = list(set(initial_recommendations))
        
        # Add additional unique exercises to meet the required number
        for exercise in unique_exercises:
            if exercise not in recommendations:
                recommendations.append(exercise)
            if len(recommendations) >= n:
                break
        
        # If there are not enough unique recommendations, randomly sample more from the filtered list
        if len(recommendations) < n:
            remaining_exercises = list(filtered_exercises['Exercise'])
            random.shuffle(remaining_exercises)
            for exercise in remaining_exercises:
                if exercise not in recommendations:
                    recommendations.append(exercise)
                if len(recommendations) >= n:
                    break
        
        # Fallback to randomly sample from injury category if needed
        if len(recommendations) < n:
            fallback_exercises = data[data['injury'].str.lower() == user_input['injury'].lower()]['Exercise'].unique()
            random.shuffle(fallback_exercises)
            for exercise in fallback_exercises:
                if exercise not in recommendations:
                    recommendations.append(exercise)
                if len(recommendations) >= n:
                    break

        return recommendations[:n]
    except Exception as e:
        print(f"Error in getting exercise recommendation: {e}")
        traceback.print_exc()
        raise

@app.post("/recommend", response_model=ExerciseRecommendation)
def recommend_exercise(user_input: UserInput):
    try:
        # Prepare user input as dictionary
        user_input_dict = {
            'age': user_input.age,
            'height': user_input.height,
            'weight': user_input.weight,
            'gender': user_input.gender,
            'injury': user_input.injury,
            'level': user_input.level,
            'equipment': user_input.equipment
        }
        
        # Get exercise recommendation
        recommended_exercises = get_exercise_recommendation(user_input_dict, rf_classifier, data, X_encoded)
        
        return ExerciseRecommendation(exercises=recommended_exercises)
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

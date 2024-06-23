from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import random

app = FastAPI()

# Load the trained model
try:
    rf_classifier = joblib.load('model.pkl')

    print('model loaded111111111111')
    X_encoded = pd.get_dummies(pd.DataFrame(columns=rf_classifier.feature_names_in_))
    # Load the data for filtering
    data = pd.read_csv('Exercises.csv')
except Exception as e:
    print(f"Error loading model or data: {e}")

class UserInput(BaseModel):
    age: int
    height: float
    weight: float
    gender: str
    injury: str
    level: str
    equipment: str

class ExerciseRecommendation(BaseModel):
    exercise: str

# Function to predict exercise
def predict_exercise(model, user_input, X_encoded):
    # Prepare the user input for prediction
    user_input['gender_encoded'] = 1 if user_input['gender'].lower() == 'female' else 0
    user_input_df = pd.DataFrame([user_input])
    user_input_encoded = pd.get_dummies(user_input_df, columns=['injury'])
    user_input_encoded = user_input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    
    # Predict exercise
    predicted_exercise = model.predict(user_input_encoded)[0]
    return predicted_exercise

# Function to filter exercises based on user specifications
def filter_exercises(data, injury, equipment, level):
    # Adjust equipment filter
    if equipment.lower() == 'dumbbells':
        equipment_filter = ['Dumbbells', 'Body Only']
    else:
        equipment_filter = [equipment.lower()]
    
    # Adjust level filter
    if level.lower() == 'beginner':
        level_filter = ['Beginner']
    elif level.lower() == 'intermediate':
        level_filter = ['Beginner', 'Intermediate']
    elif level.lower() == 'advanced':
        level_filter = ['Beginner', 'Intermediate', 'Advanced']
    else:
        level_filter = [level.lower()]
    
    filtered_df = data[(data['injury'] == injury) & (data['equipment'].str.lower().isin(equipment_filter)) & (data['level'].str.lower().isin(level_filter))]
    return filtered_df

# Main function to get recommendation
def get_exercise_recommendation(user_input, model, data, X_encoded, n=3):
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
    
    # Filter exercises based on user's injury, equipment, and level
    filtered_exercises = filter_exercises(data, user_input['injury'], user_input['equipment'], user_input['level'])
    
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
    
    # If we still don't have enough exercises, expand the search criteria
    if len(recommendations) < n:
        # Fallback: Get random exercises from the same injury category
        fallback_exercises = data[data['injury'] == user_input['injury']]['Exercise'].unique()
        random.shuffle(fallback_exercises)
        for exercise in fallback_exercises:
            if exercise not in recommendations:
                recommendations.append(exercise)
            if len(recommendations) >= n:
                break

    return recommendations[:n]

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
        
        return ExerciseRecommendation(exercise=", ".join(recommended_exercises))
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

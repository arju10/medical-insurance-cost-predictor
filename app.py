import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('best_insurance_model.pkl','rb') as f:
    model= pickle.load(f)

def predict_insurance_cost(age,sex,bmi,children,smoker,region):
    """
    Predict insurance cost based on input features
    """
    # Encode categorical variables
    sex_encoded=0 if sex=="female" else 1
    smoker_encoded=0 if smoker== "no" else 1
    
    # Map region to encoded value
    region_map= {
        "northeast": 0,
        "northwest": 1,
        "southeast": 2,
        "southwest": 3
    }
    region_encoded =region_map[region]
    
    # Create age group
    if age<=25:
        age_group = 0
    elif age<=40:
        age_group = 1
    elif age<=55:
        age_group = 2
    else:
        age_group = 3
    
    # Create input dataframe
    input_data=pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded],
        'region': [region_encoded],
        'age_group': [age_group]
    })
    
    # Make prediction
    prediction=model.predict(input_data)[0]
    
    return f"${prediction:,.2f}"

# Create Gradio interface
demo=gr.Interface(
    fn=predict_insurance_cost,
    inputs=[
        gr.Slider(18, 64, value=30,label="Age",step=1),
        gr.Radio(["female","male"],label="Sex", value="male"),
        gr.Slider(15, 55,value=25, label="BMI (Body Mass Index)",step=0.1),
        gr.Slider(0, 5,value=0,label="Number of Children", step=1),
        gr.Radio(["no", "yes"], label="Smoker", value="no"),
        gr.Radio(["northeast","northwest","southeast","southwest"], 
                label="Region", value="southwest")
    ],
    outputs=gr.Textbox(label="Predicted Insurance Cost"),
    title="Medical Insurance Cost Prediction",
    description="Enter the details below to predict medical insurance costs.",

)

if __name__ == "__main__":
    demo.launch(share=True)
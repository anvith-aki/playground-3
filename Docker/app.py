import re
import gradio
import gradio as gr
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Set the tracking URI to the server
# Provide the Forwarded address from Codespace
mlflow.set_tracking_uri("https://ominous-space-pancake-p67j45rjq9vc6p9j-5000.app.github.dev/")

# Create MLflow client
client = mlflow.tracking.MlflowClient()

# Load model from MLflow Model Registry

model_name = "loan-default-model"  # Replace with your model name
alias = "production"

# Get the model version using the alias
model_version = client.get_model_version_by_alias(name=model_name, alias=alias)

# Construct the model URI
model_uri = f"models:/{model_name}/{model_version.version}"

print(f'Model version fetched: {model_version.version}')

# Load the model using the model_uri
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

# Implementation
# Load Label encoder pickle files
profession_encoder = joblib.load("profession_encoder.pkl")
city_encoder = joblib.load("city_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")

# UI - Input components
in_Income = gradio.Textbox(lines=1, placeholder=None, value="3017006", label='Income')
in_Age = gradio.Textbox(lines=1, placeholder=None, value="32", label='Age')
in_Experience = gradio.Textbox(lines=1, placeholder=None, value="4", label='Professional experience in years')
in_Married_Single = gradio.Radio(['single', 'married'], type="value", label='Married/Single')
in_House_Ownership = gradio.Radio(['norent_noown', 'rented', 'owned'], type="value", label='House Ownership')
in_Car_Ownership = gradio.Radio(['yes', 'no'], type="value", label='Car Ownership')
in_Profession = gr.Dropdown(list(profession_encoder.classes_), label="Profession")
in_CITY = gr.Dropdown(list(city_encoder.classes_), label="City of residence")
in_STATE = gr.Dropdown(list(state_encoder.classes_), label="State of residence")
in_CURRENT_JOB_YRS = gradio.Textbox(lines=1, placeholder=None, value="4", label='Years of experience in the current job')
in_CURRENT_HOUSE_YRS = gradio.Textbox(lines=1, placeholder=None, value="11", label='Number of years in the current residence')


# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")


# Mappings for categorical features
marital_mapping = {'single': 0, 'married': 1}
house_mapping = {'norent_noown': 0, 'rented': 1, 'owned': 2}
car_mapping = {'no': 0, 'yes': 1}


# Label prediction function
def get_output_label(in_Income, in_Age, in_Experience, in_Married_Single, in_House_Ownership, in_Car_Ownership, in_Profession, in_CITY, in_STATE, in_CURRENT_JOB_YRS, in_CURRENT_HOUSE_YRS):
    try:
        input_df = pd.DataFrame({'Income': [int(in_Income)],
                                 'Age': [int(in_Age)],
                                 'Experience': [int(in_Experience)],
                                 'Married/Single': [marital_mapping[in_Married_Single]],
                                 'House_Ownership': [house_mapping[in_House_Ownership]],
                                 'Car_Ownership': [car_mapping[in_Car_Ownership]],
                                 'Profession': profession_encoder.transform([in_Profession])[0],
                                 'CITY': city_encoder.transform([in_CITY])[0],
                                 'STATE': state_encoder.transform([in_STATE])[0],
                                 'CURRENT_JOB_YRS': [int(in_CURRENT_JOB_YRS)],
                                 'CURRENT_HOUSE_YRS': [int(in_CURRENT_HOUSE_YRS)]
                                 })
    except Exception as e:
        return f"Error in input: {e}"

    try:
        prediction = loaded_model.predict(input_df)     # Make prediction using the loaded model
        if prediction[0] == 1:
            label = "Likely to Default"
        else:
            label = "Less likely Default"

        return label
    except Exception as e:
          return f"Error in model prediction: {e}"

# Create Gradio interface object
iface = gradio.Interface(fn = get_output_label,
                         inputs = [in_Income, in_Age, in_Experience, in_Married_Single, in_House_Ownership, in_Car_Ownership, in_Profession, in_CITY, in_STATE, in_CURRENT_JOB_YRS, in_CURRENT_HOUSE_YRS],
                         outputs = [out_label],
                         title="Loan Defaulter Prediction API",
                         description="Predictive model to identify possible defaulters for the consumer loans product.",
                         flagging_mode='never'
                         )


# Launch gradio interface
iface.launch(server_name = "0.0.0.0", server_port = 7860)
# set server_name = "0.0.0.0" and server_port = 7860 while launching it inside container.
# default server_name = "127.0.0.1", and server_port = 7860
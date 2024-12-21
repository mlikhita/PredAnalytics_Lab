# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:16:33 2024

@author: veenakuridi
"""

from flask import Flask, request, render_template
import pickle
import pandas as pd

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
scaler_file_x = 'pkl_files/scaler_x.pkl'
scaler_file_y = 'pkl_files/scaler_y.pkl'
model_file_final = 'pkl_files/model_final.pkl'
model_input_schema = 'pkl_files/model_input_schema.pkl'
scaler_x = pickle.load(open(scaler_file_x, 'rb'))
scaler_y = pickle.load(open(scaler_file_y, 'rb'))
model = pickle.load(open(model_file_final, 'rb'))
model_ip_schema = pickle.load(open(model_input_schema, 'rb'))

value_name = ['Inches','Ram','Weight','MemoryGB','Company','TypeName',
              'OpSys','ScreenType','TouchScreen','CpuModel','GpuBrand']

def build_model_input(ref_user_input, scaler_x, model_ip_schema):
  user_value_names = ref_user_input.keys()
  model_input_cols = model_ip_schema.keys()
  model_input = {}

  for i in user_value_names:
    if i in model_input_cols: ### covers the case of integet values
      model_input[i] = ref_user_input[i]
      # model_input.append(ref_user_input[i])
    else:
      paticular_category =[x for x in model_input_cols if i in x]
      for j in paticular_category:
        if 'TouchScreen' in j:
          flag = 0 if ref_user_input[i] == 'No' else 1
          model_input[j] = 1 if str(flag) in j else 0
        elif j.split("_")[1]==ref_user_input[i]:
          model_input[j] = True
        else:
          model_input[j] = False

  model_input_df = pd.DataFrame(data = [model_input.values()], columns = model_input.keys())
  model_input_df = model_input_df.astype(model_ip_schema)
  x_numerical_cols = ['Inches','Ram','Weight','MemoryGB']
  model_input_df[x_numerical_cols] = scaler_x.transform(model_input_df[x_numerical_cols])
  return model_input_df


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.form.values())
        inches = float(request.form['Inches'])
        ram = int(request.form['Ram'])
        memory_gb = int(request.form['MemoryGB'])
        weight = float(request.form['Weight'])
        company = request.form['Company']
        typename = request.form['TypeName']
        opsys = request.form['OpSys']
        screentype = request.form['ScreenType']
        touchscreen = int(request.form['TouchScreen'])
        cpu_model = request.form['CpuModel']
        gpu_brand = request.form['GpuBrand'] 
        user_input = [];
        user_input.append(inches); user_input.append(ram); user_input.append(weight); user_input.append(memory_gb); 
        user_input.append(company); user_input.append(typename); user_input.append(opsys); user_input.append(screentype); 
        user_input.append(touchscreen); user_input.append(cpu_model); user_input.append(gpu_brand); 
        ref_user_input = dict(zip(value_name, user_input))
        print(ref_user_input)
        features = build_model_input(ref_user_input, scaler_x, model_ip_schema)  #Convert to the form [[a, b]] for input to the model
        y_pred_rf = model.predict(features)
        result = scaler_y.inverse_transform(y_pred_rf.reshape(-1,1))[0][0]  # features Must be in the form [[a, b]]
    
        output = round(result, 2)
        print(output)

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run()
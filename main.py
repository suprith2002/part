from flask import Flask, request, render_template
import pandas as pd
import joblib
import mysql.connector
# Load the model and encoder


# Load the dataset from your database 


db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'suprith@15092002',
    'database': 'seventhmodels'
}

# Connect to the MySQL database
def get_data_from_mysql(query):
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# # Load your DataFrame from the MySQL database
data = get_data_from_mysql("SELECT * FROM model")
df=data
# Preprocess the data
oes_data = df[df['part_type'] == 'OES']
oem_data = df[df['part_type'] == 'OEM']

app = Flask(__name__)

def find_matching_parts(oem_part_number):
    
    model = joblib.load('random_forest_model.pkl')
    encoder = joblib.load('onehot_encoder.pkl')
    oem_part = oem_data[oem_data['part_number'] == oem_part_number]
    
    if oem_part.empty:
        return pd.DataFrame({"Error": ["OEM part number not found in the dataset."]})
    
    variant_key = oem_part['variant_key'].values[0]
    part_sub_category = oem_part['part_sub_category'].values[0]
    
    matching_parts = oes_data[(oes_data['variant_key'] == variant_key) & 
                              (oes_data['part_sub_category'] == part_sub_category)]
    
    if matching_parts.empty:
        return pd.DataFrame({"Error": ["No matching OES parts found."]})
    
    feature_columns = ['variant_key', 'part_sub_category']
    X_potential = matching_parts[feature_columns]
    
    X_potential_transformed = encoder.transform(X_potential)
    y_pred = model.predict(X_potential_transformed)
    
    matching_parts['Prediction'] = y_pred
    
    return matching_parts[['part_number', 'part_type', 'part_category', 'part_sub_category', 'mrp', 'brand_name']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    oem_part_number = request.form.get('oem_part_number')
    
    if oem_part_number:
        matching_parts_info = find_matching_parts(oem_part_number)
        
        if not matching_parts_info.empty:
            return matching_parts_info.to_html()
        else:
            return "No matching parts found."
    else:
        return "Please enter a valid OEM part number."

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import joblib
import mysql.connector

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




# Load the dataset from your database or a CSV file


# Preprocess the data
oes_data = df[df['part_type'] == 'OES']
oem_data = df[df['part_type'] == 'OEM']

# Combine both datasets for training
combined_data = pd.concat([oes_data, oem_data])
sample_size = 1771362  # Adjust based on memory constraints
combined_data = shuffle(combined_data).sample(n=sample_size, random_state=42)

# Define features and target
features = ['variant_key', 'part_sub_category']
target = 'part_number'

# One-hot encode the categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_features = encoder.fit_transform(combined_data[features])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_features, combined_data[target], test_size=0.2, random_state=42)

# Define and train the model
model = RandomForestClassifier(n_estimators=10, max_depth=10)
model.fit(X_train, y_train)

# Save the trained model and encoder
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(encoder, 'onehot_encoder.pkl')

print("Model and encoder saved successfully.")

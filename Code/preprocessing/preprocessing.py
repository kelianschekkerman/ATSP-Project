import pandas as pd
import re

file_path = "../../Data/original_disease_drug_data.csv"

def load_data():
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Replace / by AND
    data = re.sub(r'/', 'AND', data)

    # Remove all text between ()
    data = re.sub(r'\([^()]*\)', '', data)
    
    # For all entries with structure: disease, specification
    # Transform into (specification disease) for sentence readability

    return data

def save_data(data, columns):
    # Convert counters to DataFrame for easy saving to CSV
    df_data = pd.DataFrame(data)

    file_name = "../../Data/Preprocessed_data/preprocessed_disease_drug_data.csv"
    df_data.to_csv(file_name, index=False, columns=columns)
    print("Saved data to: " + file_name)

data = load_data()

data['disease'] = data['disease'].apply(clean_data)
data['drug'] = data['drug'].apply(clean_data)

columns = ['disease', 'drug']
save_data(data, columns)
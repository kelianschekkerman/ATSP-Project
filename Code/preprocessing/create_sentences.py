import pandas as pd
import random
import re

# Sentences file paths
file_path1 = "../../Data/Report_data/drug_name_to_disease_name.csv"
file_path2 = "../../Data/Report_data/person_name_to_drug_name.csv"
file_path3 = "../../Data/Report_data/person_name_to_disease_name.csv"

# Dataset file paths
file_path4 = "../../Data/Preprocesses_data/final/disease_drug_full.csv"
file_path5 = "../../Data/Preprocesses_data/final/name_drug_full.csv"
file_path6 = "../../Data/Preprocesses_data/final/name_disease_full.csv"

def load_data(file_path):
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def save_data(data, columns, file_name):
    # Convert counters to DataFrame for easy saving to CSV
    df_data = pd.DataFrame(data)
    df_data.to_csv(file_name, index=False, columns=columns)
    print("Saved data to: " + file_name)

# Load the relevant datasets
sentences1 = load_data(file_path1)
sentences2 = load_data(file_path2)
sentences3 = load_data(file_path3)
data1 = load_data(file_path4)
data2 = load_data(file_path5)
data3 = load_data(file_path6)

# Create new Dataframes
df = pd.DataFrame(columns = ['drug to disease'])
df2 = pd.DataFrame(columns = ['person to drug'])
df3 = pd.DataFrame(columns = ['person to disease'])

for i in data1:
    # Place the disease name and drug name in the placeholder
    result1 = re.sub(r'<disease name>', i['disease'], random.choice(sentences1))
    result2 = re.sub(r'<drug name>', i['drug'], result1)
    df = df.append({'drug to disease' : result2}, ignore_index=True)

for i in data2:
    # Place the patient name and drug name in the placeholder
    result1 = re.sub(r'<person name>', i['patient name'], random.choice(sentences2))
    result2 = re.sub(r'<drug name>', i['drug'], result1)
    df2 = df2.append({'person to drug' : result2}, ignore_index=True)

for i in data3:
    # Place the patient name in the placeholder, but leave the disease open for the LM to guess
    result1 = re.sub(r'<person name>', i['patient name'], random.choice(sentences3))
    df3 = df3.append({'person to disease' : result1}, ignore_index=True)

save_data(df, ['drug to disease'], "../../Data/Sentences/drug_to_disease_full.csv")
save_data(df2, ['person to drug'], "../../Data/Sentences/person_to_drug_full.csv")
save_data(df3, ['person to disease'], "../../Data/Sentences/person_to_disease_full.csv")
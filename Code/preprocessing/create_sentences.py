import pandas as pd
import random
import re

# Sentences file paths
file_path1 = "Data/Report_data/drug_name_to_disease_name.csv"
file_path2 = "Data/Report_data/person_name_to_drug_name.csv"
file_path3 = "Data/Report_data/person_name_to_disease_name.csv"

# Dataset file paths
file_path4 = "Data/Preprocessed_data/final/disease_drug_red10.csv"
file_path5 = "Data/Preprocessed_data/final/name_drug_red10.csv"
file_path6 = "Data/Preprocessed_data/final/name_disease_red10.csv"

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
data1_df = pd.DataFrame(data1)
data2_df = pd.DataFrame(data2)
data3_df = pd.DataFrame(data3)

df = pd.DataFrame(columns = ['drug to disease'])
df2 = pd.DataFrame(columns = ['person to drug'])
df3 = pd.DataFrame(columns = ['person to disease'])

for i in data1_df.index:
    # Place the disease name and drug name in the placeholder
    result1 = re.sub(r'<disease name>', data1_df['disease'][i], random.choice(sentences1['<drug name> to <disease name>']))
    result2 = re.sub(r'<drug name>', data1_df['drug'][i], result1)
    new_row = pd.DataFrame({'drug to disease' : result2}, index=[0])
    df = pd.concat([new_row, df.loc[:]]).reset_index(drop=True)

for i in data2_df.index:
    # Place the patient name and drug name in the placeholder
    result1 = re.sub(r'<person name>', data2_df['patient name'][i], random.choice(sentences2['<person name> to <drug name>']))
    result2 = re.sub(r'<drug name>', data2_df['drug'][i], result1)
    new_row = pd.DataFrame({'person to drug' : result2}, index=[0])
    df2 = pd.concat([new_row, df2.loc[:]]).reset_index(drop=True)

for i in data3_df.index:
    # Place the patient name in the placeholder, but leave the disease open for the LM to guess
    result1 = re.sub(r'<person name>', data3_df['patient name'][i], random.choice(sentences3['<person name> to <disease name>']))
    new_row = pd.DataFrame({'person to disease' : result1}, index=[0])
    df3 = pd.concat([new_row, df3.loc[:]]).reset_index(drop=True)

save_data(df, ['drug to disease'], "Data/Sentences/drug_to_disease_red10.csv")
save_data(df2, ['person to drug'], "Data/Sentences/person_to_drug_red10.csv")
save_data(df3, ['person to disease'], "Data/Sentences/person_to_disease_red10.csv")
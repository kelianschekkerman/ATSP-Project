import pandas as pd
import random
from tqdm import tqdm

columns = ['sentence', 'label']

# Dataset file paths
file_path1 = "../../Data/Preprocessed_data/name_disease_drug/name_disease_drug_full.csv"

# Location of new data
file_path2 = "../../Data/Sentences/prompts/name_disease_prompt_simple_full.csv"
file_path3 = "../../Data/Sentences/prompts/name_drug_prompt_simple_full.csv"
file_path4 = "../../Data/Sentences/prompts/disease_name_prompt_simple_full.csv"
file_path5 = "../../Data/Sentences/prompts/disease_drug_prompt_simple_full.csv"
file_path6 = "../../Data/Sentences/prompts/drug_disease_prompt_simple_full.csv"
file_path7 = "../../Data/Sentences/prompts/drug_name_prompt_simple_full.csv"

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
data = load_data(file_path1)

# Create new Dataframes
df = pd.DataFrame(data)

df1 = pd.DataFrame(columns= columns)
df2 = pd.DataFrame(columns= columns)
df3 = pd.DataFrame(columns= columns)
df4 = pd.DataFrame(columns= columns)
df5 = pd.DataFrame(columns= columns)
df6 = pd.DataFrame(columns= columns)

for i in tqdm(df.index, desc="Processing rows", unit="row"):
    # df1 name - disease
    new_row = [df['patient name'][i] + " suffers from disease <MASK>", df['disease'][i]]
    df1.loc[len(df1)] = new_row
    # df2 name - drug
    new_row = [df['patient name'][i] + " takes drug <MASK>", df['drug'][i]]
    df2.loc[len(df2)] = new_row
    # df3 disease - name
    new_row = [df['disease'][i] + " has impacted patient by the name of <MASK>", df['patient name'][i]]
    df3.loc[len(df3)] = new_row
    # df4 disease - drug
    new_row = [df['disease'][i] + " is treated with drug <MASK>", df['drug'][i]]
    df4.loc[len(df4)] = new_row
    # df5 drug - disease
    new_row = [df['drug'][i] + " is taken for disease <MASK>", df['disease'][i]]
    df5.loc[len(df5)] = new_row
    # df6 drug - name
    new_row = [df['drug'][i] + " is prescribed for patient by the name of <MASK>", df['patient name'][i]]
    df6.loc[len(df6)] = new_row

save_data(df1, columns, file_path2)
save_data(df2, columns, file_path3)
save_data(df3, columns, file_path4)
save_data(df4, columns, file_path5)
save_data(df5, columns, file_path6)
save_data(df6, columns, file_path7)
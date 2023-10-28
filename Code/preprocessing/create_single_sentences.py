import pandas as pd
from tqdm import tqdm

# Dataset file paths
file_path1 = "../../Data/Preprocessed_data/name_disease_drug/name_disease_drug_reduced10.csv"

# Single sentence locations
file_path5 = "../../Data/Sentences/single/name_single_red10.csv"
file_path6 = "../../Data/Sentences/single/disease_single_red10.csv"
file_path7 = "../../Data/Sentences/single/drug_single_red10.csv"
file_path8 = "../../Data/Sentences/single/all_single_red10.csv"

def load_data(file_path):
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def save_data(data, columns, file_name):
    # Convert counters to DataFrame for easy saving to CSV
    df_data = pd.DataFrame(data)
    df_data.to_csv(file_name, index=False, columns=columns)
    print("Saved data to: " + file_name)

data1 = load_data(file_path1)

df = pd.DataFrame(data1)
df1 = pd.DataFrame(columns= ['patient name'])
df2 = pd.DataFrame(columns= ['disease'])
df3 = pd.DataFrame(columns= ['drug'])

for i in tqdm(df.index, desc="Processing rows", unit="row"):
    # df1 patient name
    new_row = [df['patient name'][i] + " is a patient name."]
    df1.loc[len(df1)] = new_row
    # df2 disease
    new_row = [df['disease'][i] + " is a disease."]
    df2.loc[len(df2)] = new_row
    # df1 drug
    new_row = [df['drug'][i] + " is a drug."]
    df3.loc[len(df3)] = new_row

save_data(df1, ['patient name'], file_path5)
save_data(df2, ['disease'], file_path6)
save_data(df3, ['drug'], file_path7)

# From split datasets create one dataset called all
df_all = [df1, df2, df3]
result = pd.concat(df_all, axis=1, join='inner')
save_data(result, ['patient name', 'disease', 'drug'], file_path8)

# Turn into text
csv_file = "../../Data/Sentences/single/all_single_red10.csv"
txt_file = "../../Data/Sentences/single/all_single_red10.txt"
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(row) for row in my_input_file]
    my_output_file.close()
import pandas as pd

# Dataset file paths
file_path1 = "../../Data/Preprocessed_data/disease_drug/disease_drug_data_full.csv"
file_path2 = "../../Data/Preprocessed_data/names/full_unique_names.csv"

# Seperate dataset locations
file_path3 = "../../Data/Preprocessed_data/disease_drug/disease.csv"
file_path4 = "../../Data/Preprocessed_data/disease_drug/drug.csv"

# Single sentence locations
file_path5 = "../../Data/Sentences/single/disease_single.csv"
file_path6 = "../../Data/Sentences/single/drug_single.csv"
file_path7 = "../../Data/Sentences/single/name_single.csv"
file_path8 = "../../Data/Sentences/single/all_single.csv"

def load_data(file_path):
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def save_data(data, columns, file_name):
    # Convert counters to DataFrame for easy saving to CSV
    df_data = pd.DataFrame(data)
    df_data.to_csv(file_name, index=False, columns=columns)
    print("Saved data to: " + file_name)

# data1 = load_data(file_path1)
# data2 = load_data(file_path2)

# df_disease = pd.DataFrame(data1['disease'])
# df_drug = pd.DataFrame(data1['drug'])
# df_name = pd.DataFrame(data2)

# save_data(df_disease, ['disease'], file_path3)
# save_data(df_drug, ['drug'], file_path4)

# for i in df_disease.index:
#     df_disease['disease'][i] = df_disease['disease'][i] + " is a disease."

# for i in df_drug.index:
#     df_drug['drug'][i] = df_drug['drug'][i] + " is a drug."

# for i in df_name.index:
#     df_name['patient name'][i] = df_name['patient name'][i] + " is a patient name."

# save_data(df_disease, ['disease'], file_path5)
# save_data(df_drug, ['drug'], file_path6)
# save_data(df_name, ['patient name'], file_path7)


# disease = load_data(file_path5)
# drug = load_data(file_path6)
# name = load_data(file_path7)

# df_disease = pd.DataFrame(disease)
# df_drug = pd.DataFrame(drug)
# df_name = pd.DataFrame(name)

# df_all = [df_name, df_disease, df_drug]
# result = pd.concat(df_all, axis=1, join='inner')
# save_data(result, ['patient name', 'disease', 'drug'], file_path8)


csv_file = "../../Data/Sentences/single/all_single.csv"
txt_file = "../../Data/Sentences/single/all_single.txt"
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(row) for row in my_input_file]
    my_output_file.close()
import pandas as pd

file_path1 = "../../Data/Preprocessed_data/names/unique_names.csv"
file_path2 = "../../Data/Preprocessed_data/disease_drug/disease_drug_reduced10.csv" # Run it for original, reduced 3, 5, and 10

# Load in the csv file with the wanted columns
def load_data(file_path):
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def save_data(data, columns, path):
    data.to_csv(path, index=False, columns=columns)
    print("Saved data to: " + path)

# Combine the first and last name.
data = load_data(file_path1)
data['patient name'] = data['First Name'] + " " + data['Last Name']

# Load in the dataset
# data2 = load_data(file_path2)
df = pd.DataFrame(data)

# # Add the Full Name column to the dataset
# dataLength = len(df)
# data = data[:dataLength]
# df['patient name'] = data['patient name']

# columns = ['patient name', 'disease', 'drug']
save_data(df, ['patient name'], "../../Data/Preprocessed_data/names/full_uniques_names.csv")
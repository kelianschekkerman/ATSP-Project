import pandas as pd

# Load in the csv file with the wanted columns
def load_data(columnName1, columnName2, file_path):
    print("Importing data...")
    columns = [columnName1, columnName2]      # Import only these columns
    data = pd.read_csv(file_path, usecols=columns)
    return data 

# Combine the first and last name.
data = load_data('First Name', 'Last Name', "Data/names.csv")
data['Full Name'] = data['First Name'] + " " + data['Last Name']

# Load in the dataset
data2 = load_data('disease', 'drug', "Data/original_disease_drug_data.csv")
df = pd.DataFrame(data2)

# Add the Full Name column to the dataset
dataLength = len(df)
data = data[:dataLength]
df['Full Name'] = data['Full Name']

print(df)
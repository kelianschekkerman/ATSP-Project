import pandas as pd

file_path1 = "../../Data/unique_names.csv"
file_path2 = "../../Data/reduced3_disease_drug_data.csv" # Run it for original, reduced 3 to reduced 5, and reduced 10

# Load in the csv file with the wanted columns
def load_data(columnName1, columnName2, file_path):
    print("Importing data...")
    columns = [columnName1, columnName2]      # Import only these columns
    data = pd.read_csv(file_path, usecols=columns)
    return data

def save_data(data, columns):
    # Convert to DataFrame for easy saving to CSV
    df_data = pd.DataFrame(data)

    path = "../../Data/preprocessed_name_disease_drug.csv"
    df_data.to_csv(path, index=False, columns=columns)
    print("Saved data to: " + path)

# Combine the first and last name.
name_data = load_data('First Name', 'Last Name', file_path1)
name_data['Full Name'] = name_data['First Name'] + " " + name_data['Last Name']

# Load in the dataset
dd_data = load_data('disease', 'drug', file_path2)
df = pd.DataFrame(dd_data)

# Add the Full Name column to the dataset
dataLength = len(df)
data = dd_data[:dataLength]
df['Full Name'] = data['Full Name']

save_data(df)
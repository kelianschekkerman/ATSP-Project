import pandas as pd

file_path = "../../Data/preprocessed_name_disease_drug.csv"

def load_data():
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def save_data(data, columns, path):
    # Convert to DataFrame for easy saving to CSV
    df_data = pd.DataFrame(data)

    df_data.to_csv(path, index=False, columns=columns)
    print("Saved data to: " + path)

data = load_data()

# Save the three separate datasets
save_data(data, ['Full Name', 'disease'], "../../Data/Training_data/name_disease.csv")  # name <-> disease dataset
save_data(data, ['Full Name', 'drug'], "../../Data/Training_data/name_drug.csv")        # name <-> drug dataset
save_data(data, ['disease', 'drug'], "../../Data/Training_data/disease_drug.csv")       # disease <-> drug dataset
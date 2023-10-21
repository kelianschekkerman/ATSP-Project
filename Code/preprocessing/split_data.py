import pandas as pd

file_path = "../../Data/Preprocessed_data/name_disease_drug_reduced10.csv"

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
save_data(data, ['patient name', 'disease'], "../../Data/Preprocessed_data/final/name_disease_red10.csv")  # name <-> disease dataset
save_data(data, ['patient name', 'drug'], "../../Data/Preprocessed_data/final/name_drug_red10.csv")        # name <-> drug dataset
save_data(data, ['disease', 'drug'], "../../Data/Preprocessed_data/final/disease_drug_red10.csv")          # disease <-> drug dataset
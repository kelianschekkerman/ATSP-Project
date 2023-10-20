import pandas as pd
from collections import Counter

file_path = "../../Data/original_disease_drug_data.csv"

def load_data():
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def count_frequencies(data):
    print("Counting...")

    # Get the necessary columns from the data and transform it into a list
    diseases = data['disease'].to_list()
    drugs = data['drug'].to_list()

    # Count the frequency of each entry
    disease_frequency = Counter(diseases)
    drug_frequency = Counter(drugs)

    # Return the frequencies of both lists in descending order
    return sorted(disease_frequency.items(), key = lambda x:x[1], reverse = True), sorted(drug_frequency.items(), key = lambda x:x[1], reverse = True)

def save_data(data, path):
    # Convert counters to DataFrame for easy saving to CSV
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(path, index=False)

    print("Saved data to " + path)

# Load the dataset
data = load_data()

# Count the frequencies of data entries
disease_frq, drug_frq = count_frequencies(data)

# Save the counted frequencies
save_data(disease_frq, "../../Data/disease_frequency.csv")
save_data(drug_frq, "../../Data/drug_frequency.csv")

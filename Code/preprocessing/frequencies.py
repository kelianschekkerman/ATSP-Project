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
    combo_list = data['combination'].to_list()

    # Count the frequency of each entry
    frequency = Counter(combo_list)

    # Return the frequencies of both lists in descending order
    return sorted(frequency.items(), key = lambda x:x[1], reverse = True)

def save_data(data, columns):
    # Convert counters to DataFrame for easy saving to CSV
    # df_frq = pd.DataFrame(frq)
    df_data = pd.DataFrame(data)

    # Save to CSV
    # df_frq.to_csv("../../Data/combination_frequencies.csv", index=False)
    # print("Saved data to: ../../Data/combination_frequencies.csv" )

    df_data.to_csv("../../Data/reduced_disease_drug_data.csv", index=False, columns=columns)
    print("Saved data to: ../../Data/reduced_disease_drug_data.csv" )

# Load the dataset
data = load_data()

# Construct the disease/drug combination for each entry
data['combination'] = data['disease'] + "_" + data['drug']

# Count the frequencies of disease/drug combinations
frq = count_frequencies(data)
# save_data(frq)

# Remove rows with uncommon combinations
threshold = 3       # TODO: decide on a appropriate threshold
common_combos = set(item[0] for item in frq if item[1] >= threshold)
data = data[data['combination'].isin(common_combos)]

# Save the updated data
columns = ['disease', 'drug']
save_data(data, columns)

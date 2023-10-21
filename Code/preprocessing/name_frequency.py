import pandas as pd
from collections import Counter

file_path = "../../Data/original_names_data.csv" 

def load_data():
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

def count_frequencies(data):
    print("Counting...")

    # Get the necessary columns from the data and transform it into a list
    first_names = data['First Name'].to_list()
    last_names = data['Last Name'].to_list()

    # Count the frequency of each entry
    first_names_frequency = Counter(first_names)
    last_names_frequency = Counter(last_names)

    # Return the frequencies of both lists in descending order
    return sorted(first_names_frequency.items(), key = lambda x:x[1], reverse = True), sorted(last_names_frequency.items(), key = lambda x:x[1], reverse = True)

def save_data(data, path):
    # Convert counters to DataFrame for easy saving to CSV
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(path, index=False)

    print("Saved data to " + path)

# Load the dataset
data = load_data()

# Count the frequencies of data entries
first_name_frq, last_name_frq = count_frequencies(data)

print(first_name_frq)
print(last_name_frq)

# Save the counted frequencies
# save_data(first_name_frq, "../../Data/first_name_frequency.csv")
# save_data(last_name_frq, "../../Data/last_name_frequency.csv")
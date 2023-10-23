import csv
import os
from pathlib import Path

# Read the CSV file
base_path = Path('Data') / 'Preprocessed_data' / 'name_disease_drug'

filename = base_path / 'name_disease_drug_reduced10.csv'
data = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    # Skipping the headers
    next(csvreader)
    
    # Extracting data
    for row in csvreader:
        data.append({
            "patient_name": row[0],
            "disease": row[1],
            "drug": row[2]
        })

# Generate the first raw text file
with open('merge.txt', 'w') as txtfile:
    for record in data:
        txtfile.write(f"Patient {record['patient_name']} takes drug {record['drug']}.\n")

# Generate the second raw text file
unique_diseases = set([(record['disease'], record['drug']) for record in data])
with open('merge.txt', 'a') as txtfile:
    for disease, drug in unique_diseases:
        txtfile.write(f"Drug {drug} is prescribed for disease {disease}.\n")

import os
import json
import csv

# Initialize a dictionary to store the aggregated results
results = {}
pre_train_versions = ['pre_gpt2', 'pre_gpt2-large', 'pre_roberta-base']
import os
import json
import csv

current_directory = os.getcwd()
all_folders = [f for f in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, f)) and (f not in pre_train_versions) ]

# Mapping of keys in json to headers in csv
keys_to_headers = {
    "percentage_found_in_dataset": "percentage_found_in_dataset",
    "full_string_top_1": "full_string.top_1",
    "full_string_top_3": "full_string.top_3",
    "full_string_top_5": "full_string.top_5",
    "full_string_top_10": "full_string.top_10",
    "first_three_chars_top_1": "first_three_chars.top_1",
    "first_three_chars_top_3": "first_three_chars.top_3",
    "first_three_chars_top_5": "first_three_chars.top_5",
    "first_three_chars_top_10": "first_three_chars.top_10",
}

processed_files = set()

for folder in all_folders:
    for eval_file in [f for f in os.listdir(folder) if f.startswith('eval_')]:
        if eval_file not in processed_files:
            processed_files.add(eval_file)
            results = {}
            for inner_folder in all_folders:
                full_path = os.path.join(inner_folder, eval_file)
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        results[inner_folder] = {
                            "percentage_found_in_dataset": round(data.get("percentage_found_in_dataset", 0), 2),
                            "full_string_top_1": round(data["full_string"].get("top_1", 0)*100, 2),
                            "full_string_top_3": round(data["full_string"].get("top_3", 0)*100, 2),
                            "full_string_top_5": round(data["full_string"].get("top_5", 0)*100, 2),
                            "full_string_top_10": round(data["full_string"].get("top_10", 0)*100, 2),
                            "first_three_chars_top_1": round(data["first_three_chars"].get("top_1", 0)*100, 2),
                            "first_three_chars_top_3": round(data["first_three_chars"].get("top_3", 0)*100, 2),
                            "first_three_chars_top_5": round(data["first_three_chars"].get("top_5", 0)*100, 2),
                            "first_three_chars_top_10": round(data["first_three_chars"].get("top_10", 0)*100, 2),
                        }

            with open(eval_file.replace('.json', '.csv'), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['Metric'] + all_folders)
                writer.writeheader()
                for key, header in keys_to_headers.items():
                    row_data = {"Metric": key}
                    for folder_name, folder_data in results.items():
                        row_data[folder_name] = folder_data[key]
                    writer.writerow(row_data)

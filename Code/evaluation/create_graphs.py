import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sentences file paths
file_path = "results/merged/eval_disease_name_prompt_simple_red10.csv"

def load_data(file_path):
    print("Importing data...")
    data = pd.read_csv(file_path)
    return data

plt.rcParams["figure.figsize"] = [7.5, 5]
plt.rcParams["figure.autolayout"] = True

data = load_data(file_path)

# Make graph for gpt2-large accuracy - predictions
x = data['Metric'][1:]
y = data['gpt2-large_1_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2-large_1_simple")
y = data['gpt2-large_1_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2-large_1_variant")
y = data['gpt2-large_4_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2-large_4_simple")
y = data['gpt2-large_4_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2-large_4_variant")
y = data['gpt2-large_8_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2-large_8_simple")
y = data['gpt2-large_8_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2-large_8_variant")

plt.xticks(rotation = 25)
plt.xlabel('Predictions') 
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/graphs/name_drug_accuracy_against_number_of_predictions_gpt2-large.png")
plt.cla()

# Make graph for gpt2 accuracy - predictions
x = data['Metric'][1:]
y = data['gpt2_1_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2_1_simple")
y = data['gpt2_1_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2_1_variant")
y = data['gpt2_4_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2_4_simple")
y = data['gpt2_4_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2_4_variant")
y = data['gpt2_8_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2_8_simple")
y = data['gpt2_8_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "gpt2_8_variant")

plt.xticks(rotation = 25)
plt.xlabel('Predictions') 
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/graphs/name_drug_accuracy_against_number_of_predictions_gpt2.png")
plt.cla()

# Make graph for pre-gpt2-large accuracy - predictions
x = data['Metric'][1:]
y = data['pre_gpt2-large_PRE__1_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2-large_1_simple")
y = data['pre_gpt2-large_PRE__1_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2-large_1_variant")
y = data['pre_gpt2-large_PRE__4_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2-large_4_simple")
y = data['pre_gpt2-large_PRE__4_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2-large_4_variant")
y = data['pre_gpt2-large_PRE__8_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2-large_8_simple")
y = data['pre_gpt2-large_PRE__8_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2-large_8_variant")

plt.xticks(rotation = 25)
plt.xlabel('Predictions') 
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/graphs/name_drug_accuracy_against_number_of_predictions_pre-gpt2-large.png")
plt.cla()

# Make graph for pre-gpt2 accuracy - predictions
x = data['Metric'][1:]
y = data['pre_gpt2_PRE__1_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2_1_simple")
y = data['pre_gpt2_PRE__1_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2_1_variant")
y = data['pre_gpt2_PRE__4_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2_4_simple")
y = data['pre_gpt2_PRE__4_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2_4_variant")
y = data['pre_gpt2_PRE__8_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2_8_simple")
y = data['pre_gpt2_PRE__8_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-gpt2_8_variant")

plt.xticks(rotation = 25)
plt.xlabel('Predictions') 
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/graphs/name_drug_accuracy_against_number_of_predictions_pre-gpt2.png")
plt.cla()

# Make graph for pre-roberta accuracy - predictions
x = data['Metric'][1:]
y = data['pre_roberta-base_PRE__1_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-roberta_1_simple")
y = data['pre_roberta-base_PRE__1_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-roberta_1_variant")
y = data['pre_roberta-base_PRE__4_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-roberta_4_simple")
y = data['pre_roberta-base_PRE__4_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-roberta_4_variant")
y = data['pre_roberta-base_PRE__8_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-roberta_8_simple")
y = data['pre_roberta-base_PRE__8_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "pre-roberta_8_variant")

plt.xticks(rotation = 25)
plt.xlabel('Predictions') 
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/graphs/name_drug_accuracy_against_number_of_predictions_pre-roberta.png")
plt.cla()

# Make graph for roberta accuracy - predictions
x = data['Metric'][1:]
y = data['roberta-base_1_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "roberta_1_simple")
y = data['roberta-base_1_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "roberta_1_variant")
y = data['roberta-base_4_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "roberta_4_simple")
y = data['roberta-base_4_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "roberta_4_variant")
y = data['roberta-base_8_simple_merge_red10'][1:].tolist()
plt.plot(x, y, label = "roberta_8_simple")
y = data['roberta-base_8_variant_merge_red10'][1:].tolist()
plt.plot(x, y, label = "roberta_8_variant")

plt.xticks(rotation = 25)
plt.xlabel('Predictions') 
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/graphs/name_drug_accuracy_against_number_of_predictions_roberta.png")
plt.cla()

# Make graph for percentage found in dataset
plt.rcParams["figure.figsize"] = [14, 6]
plt.rcParams["figure.autolayout"] = True

x = data.columns.tolist()
x.remove('Metric')
x.remove('roberta-base_10_merge_red10')
y = data.loc[0, :].values.flatten().tolist()
y.remove('percentage_found_in_dataset')
y = [z for z in y if str(z) != 'nan']
plt.plot(x, y, label = "percentage_found_in_dataset")

plt.xticks(rotation = 90)
plt.xlabel('Config') 
plt.ylabel('Percentage')
plt.legend()
plt.savefig("results/graphs/disease-name_percentage_dataset.png")
plt.cla()
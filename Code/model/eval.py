from sklearn.metrics import *
import numpy as np
import json

def eval_predictions(eval_data_path, predictions, labels, output_dir, operator, description):
    # Calculate the accuracy
    acc = accuracy_score(labels, predictions)

    # Calculate the precision
    precision = precision_score(labels, predictions)

    # Calculate the recall
    recall = recall_score(labels, predictions)

    # Calculate the precision @k
    k_set = [1,3,5,10]      # Set the value for k
    p_at_k = precision_at_k(labels, predictions, k_set)

    # Calculate the F1-score
    F1 = f1_score(labels, predictions)

    # Format to save to file
    results = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'Precision@1': p_at_k[0],
        'Precision@3': p_at_k[1],
        'Precision@5': p_at_k[2],
        'Precision@10': p_at_k[3],
        'F1-Score': F1
    }

    save_to_file(results, description, eval_data_path, output_dir, operator)

# Precision at k assuming binary relevance levels
# Implementation from: https://gist.github.com/mblondel/7337391
def precision_at_k(labels, predictions, k_set):
    unique_y = np.unique(labels)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(labels == pos_label)

    order = np.argsort(predictions)[::-1]

    # Initialise a list to save the predictions
    precisions_at_k = []

    # Calculate the precision for each k in the set
    for k in k_set:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        
        top_k_truth = np.take(labels, order[:k])
        n_relevant = np.sum(top_k_truth == pos_label)

        # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
        prec_k = float(n_relevant) / min(n_pos, k)
        precisions_at_k.append(prec_k)

    return precisions_at_k

def save_to_file(results, description, eval_data_path, output_dir, operation):
    # save the results as json file in the following file
    output_path = output_dir / f"eval_{eval_data_path.stem}.json"

    print()
    print("start evaluation".upper().center(40, '='))
    print(f">>> Will save at: {output_dir}")
    print()

    with open(output_path, operation) as file:
        data = {
            'description': description,
            'results': results
        }
        json.dump(data, file, indent=4)

    

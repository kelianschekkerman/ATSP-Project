from sklearn.metrics import *
import numpy as np

filename = "../../Results/evaluation.py"

# Evaluate the performance of the predictions using several performance metrics
def evaluate_predictions(truth, predictions):
    # Calculate the accuracy
    acc = accuracy_score(truth, predictions)

    # Calculate the precision
    precision = precision_score(truth, predictions)

    # Calculate the recall
    recall = recall_score(truth, predictions)

    # Calculate the precision @k
    k = 10      # Set the value for k
    p_at_k = precision_at_k(truth, predictions, k)

    # Calculate the F1-score
    F1 = f1_score(truth, predictions)

    # Format to save to file
    results = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'Precision@10': p_at_k,
        'F1-Score': F1
    }

    return results

# Precision at k assuming binary relevance levels
# Implementation from: https://gist.github.com/mblondel/7337391
def precision_at_k(truth, predictions, k):
    unique_y = np.unique(truth)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(truth == pos_label)

    order = np.argsort(predictions)[::-1]
    truth = np.take(truth, order[:k])
    n_relevant = np.sum(truth == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)

# Write the results to a text file
def save_to_file(results, filename, operation, description):
    with open(filename, operation) as file:
        file.write(description + "\n")  # Write description
        file.write("-" * 30 + "\n")     # Add a separator

        # Write each metric and corresponding value
        for metric, value in results.items():
            file.write(f"{metric}: {value}\n")

    file.write("-" * 30 + "\n")         # Add a separator

# TODO: Import lists for truth and predictions
truth = "import this"
predictions = "import this"

# Evaluate performance of the predictions
results = evaluate_predictions(truth, predictions)

# Save results to file
save_to_file(results, filename, 'w', "Performance metrics of predictions")

# Consider only the first n characters
n = 3           # Set the value for n 
trunc_pred = [pred[:n] for pred in predictions]     # truncate predictions
trunc_truth = [true[:n] for true in truth]          # truncate truth
results_trunc = evaluate_predictions(trunc_truth, trunc_pred)       # Evaluate again with truncated data

# Save results to a text file
save_to_file(results_trunc, filename, 'a', "Performance metrics for the first 3 characters of predictions")
import json

def eval_predictions(eval_data_path, predictions, labels, output_dir):
    import os

    def accuracy_at_k(preds, lbl, k):
        """Calculate accuracy at top k."""
        return sum(1 for p, l in zip(preds, lbl) if l in p[:k]) / len(lbl)

    # Condition checks
    def full_string(pred, label):
        # only considering the first two words
        return [p.split()[:2].lower() for p in pred], label.split()[:2].lower()

    def first_word(pred, label):
        # only considering the first words
        return [p.split()[0].lower() for p in pred], label.split()[0].lower()

    def first_three_chars(pred, label):
        # only considering the first three chars
        return [p[:3].lower() for p in pred], label[:3].lower()

    conditions = [full_string, first_word, first_three_chars]

    results = {}
    for condition in conditions:
        condition_name = condition.__name__
        results[condition_name] = {}
        for k in [1, 3, 5, 10]:
            modified_predictions = [condition(pred, lbl)[0] for pred, lbl in zip(predictions, labels)]
            modified_labels = [condition(pred, lbl)[1] for pred, lbl in zip(predictions, labels)]
            results[condition_name][f'top_{k}'] = accuracy_at_k(modified_predictions, modified_labels, k)

    # Save results to file
    output_file_path = output_dir/ f"eval_{eval_data_path.stem}.json"
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

import json

def eval_predictions(eval_data_path, predictions, labels, output_dir):
    import os

    def accuracy_at_k(preds, lbl, k):
        """Calculate accuracy at top k."""
        return sum(1 for p, l in zip(preds, lbl) if l in p[:k]) / len(lbl)

    # Condition checks
    def full_string(pred, label):
        # only considering the first two words
        return [p.lower().split()[:2] for p in pred], label.lower().split()[:2]

    def first_word(pred, label):
        # only considering the first words
        return [p.lower().split()[0] if p.strip() else p for p in pred], label.lower().split()[0]

    def first_three_chars(pred, label):
        # only considering the first three chars
        return [p[:3].lower() for p in pred], label[:3].lower()
    
    def in_dataset(pred, label):
        # check if prediction is in dataset
        # return tuple: list of bools, none because labels are not used
        return [
            any(
                any(
                    p.lower() == str(item).lower()
                    for item in label
                )
                for p in pred_item
            )
            for pred_item in pred
        ], None

    conditions = [full_string, first_word, first_three_chars]

    results = {}
    pred_in_data, _ = in_dataset(predictions, labels)
    results['percentage_found_in_dataset'] = (sum(pred_in_data) / len(pred_in_data)) * 100
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

import csv
from pathlib import Path

def myprint(s):
    print(s.upper().center(50, '='))


def save_predictions(eval_data_path, predictions, labels, output_dir, complete_predictions=None):
    output_path = output_dir / f"pred_{eval_data_path.stem}.csv"
    predictions_count = len(predictions[0])
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if complete_predictions:
            # Write the header (optional)
            writer.writerow(["label"] + [f'w{i}' for i in range(predictions_count)] + [f's{i}' for i in range(predictions_count)])
        
            # Write the data
            for label, prediction_list, complete_prediction_list in zip(labels, predictions, complete_predictions):
                writer.writerow([label] + prediction_list + complete_prediction_list)
        else:
            writer.writerow(["label"] + [f'w{i}' for i in range(predictions_count)])
        
            # Write the data
            for label, prediction_list, complete_prediction_list in zip(labels, predictions):
                writer.writerow([label] + prediction_list)

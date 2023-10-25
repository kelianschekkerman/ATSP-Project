import csv

def myprint(s):
    print(s.upper().center(50, '='))


def save_predictions(eval_data_path, predictions, labels, output_dir):
    output_path = output_dir / f"pred_{eval_data_path.stem}.csv"
    predictions_count = len(predictions[0])
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header (optional)
        writer.writerow(["label"] + [str(i) for i in range(predictions_count)])
    
        # Write the data
        for label, prediction_list in zip(labels, predictions):
            writer.writerow([label] + prediction_list)

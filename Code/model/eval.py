
def eval_predictions(eval_data_path, predictions, labels, output_dir):
    # do the evaluations
    # save the results as json file in the following file
    output_path = output_dir / f"eval_{eval_data_path.stem}.json"

    print()
    print("start evaluation".upper().center(40, '='))
    print(f">>> Will save at: {output_dir}")
    print()
    

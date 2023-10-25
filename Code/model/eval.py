
def eval_predictions(eval_data_path, predictions, labels, output_dir):
    # do the evaluations
    print()
    print("start training".upper().center(40, '='))
    print(f">>> Train Dataset: {train_data_path}")
    print(f">>> Eval Dataset: {eval_data_path}")
    print(f">>> Model: {model_name or model_path}")
    print(f">>> Will save at: {output_dir}")
    print()
    
    # save the results as json file in the following file
    output_path = output_dir / f"eval_{eval_data_path.stem}.json"
    
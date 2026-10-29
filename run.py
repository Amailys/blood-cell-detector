import config
from pathlib import Path
from train import train_model
from val import validate_model, generate_mean_report


def main():
    exp_dir = config.BASE_RESULTS_DIR / config.EXP_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting experiment: '{config.EXP_NAME}'")
    print(f"Results will be saved to: {exp_dir}")

    all_run_results = []
    class_names = None

    for seed in config.SEEDS:
        print("\n" + "=" * 80)
        print(f"Processing seed: {seed}")
        print("=" * 80)

        run_dir = exp_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            best_model_path = train_model(
                model_name=config.MODEL_NAME,
                data_yaml_path=config.DATA_YAML_PATH,
                run_dir=run_dir,
                seed=seed,
                train_params=config.TRAIN_PARAMS
            )
        except Exception as e:
            print(f"Critical error during training for seed {seed}: {e}")
            continue

        if not best_model_path or not best_model_path.exists():
            print(f"Model for seed {seed} was not found. Skipping validation.")
            continue

        print(f"Starting validation for seed {seed}...")
        try:
            results, names = validate_model(
                model_path=best_model_path,
                val_dir=run_dir,
                data_yaml_path=config.DATA_YAML_PATH,
                val_images_path=config.VAL_IMAGES_PATH,
                val_labels_path=config.VAL_LABELS_PATH,
                seed=seed,
                val_params=config.VAL_PARAMS
            )

            if results:
                all_run_results.append(results)
                if class_names is None and names:
                    class_names = names
            print(f"Validation completed for seed {seed}.")

        except Exception as e:
            print(f"Critical error during validation for seed {seed}: {e}")
            continue

    print("\n" + "=" * 80)
    print("All runs finished")
    print("=" * 80)

    if class_names and len(all_run_results) >= 2:
        print(f"{len(all_run_results)} successful runs. Generating final averaged report...")
        generate_mean_report(all_run_results, class_names, exp_dir)
    else:
        print("\nAveraged report was not generated.")
        if not class_names:
            print("Reason: No validation could be completed.")
        else:
            print(f"Reason: Only {len(all_run_results)} run(s) succeeded. At least 2 are required for an averaged report.")


if __name__ == "__main__":
    main()

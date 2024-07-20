import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
    # "test_regression_model"
]

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/basic_cleaning",
                "main",
                version='main',
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "filter_column": config["basic_cleaning"]["filter_column"],
                    "filter_value": config["basic_cleaning"]["filter_value"],
                    "output_artifact": config["basic_cleaning"]["output_artifact"],
                    "output_type": config["basic_cleaning"]["output_type"],
                    "output_description": config["basic_cleaning"]["output_description"]
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_check",
                "main",
                version='main',
                parameters={
                    "input_artifact": f"{config['basic_cleaning']['output_artifact']}:latest",
                    "ref_artifact": config["data_check"]["ref_artifact"],
                    "kl_threshold": config["data_check"]["kl_threshold"]
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_split",
                "main",
                version='main',
                parameters={
                    "input_artifact": f"{config['basic_cleaning']['output_artifact']}:latest",
                    "test_size": config["data_split"]["test_size"],
                    "random_seed": config["data_split"]["random_seed"],
                    "stratify_by": config["data_split"]["stratify_by"]
                },
            )

        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_random_forest",
                "main",
                version='main',
                parameters={
                    "train_data": "train.csv:latest",
                    "validation_data": "validation.csv:latest",
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": config["modeling"]["output_artifact"],
                    "output_type": config["modeling"]["output_type"],
                    "output_description": config["modeling"]["output_description"]
                },
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                version='main',
                parameters={
                    "mlflow_model": f"{config['modeling']['output_artifact']}:prod",
                    "test_data": "test.csv:latest"
                },
            )

if __name__ == "__main__":
    go()
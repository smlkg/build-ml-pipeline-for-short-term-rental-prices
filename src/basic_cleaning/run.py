#!/usr/bin/env python
"""
data cleaning
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Reading data from {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    # Basic data cleaning steps
    logger.info("Cleaning data...")
    
    # Example of basic data cleaning
    df = df.drop_duplicates()  # Drop duplicate rows
    df = df.dropna()           # Drop rows with missing values
    
    # Assuming the cleaning involves filtering out rows based on certain conditions
    df = df[df[args.filter_column] >= args.filter_value]

    # Save the cleaned data to a new CSV file
    output_path = args.output_artifact
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")

    # Log the output artifact
    run.log_artifact(
        wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description
        )
    )
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact to clean",
        required=True
    )

    parser.add_argument(
        "--filter_column", 
        type=str,
        help="Column to filter on",
        required=True
    )

    parser.add_argument(
        "--filter_value", 
        type=float,
        help="Value to filter the column on",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    args = parser.parse_args()

    go(args)
import subprocess
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run data processing script")
    parser.add_argument("--augmentation_factor", type=int, default=5, help="Factor by which to augment the dataset")
    args = parser.parse_args()

    # Run data processing script
    print('==='*30)
    print(f'Running data processing with augmentation factor: {args.augmentation_factor}')
    subprocess.run(["python", "data_processing.py", "--augmentation_factor", str(args.augmentation_factor)], check=True)
    print('==='*30)
    print('\n')

    # Run training script
    print('==='*30)
    print("Starting training...")
    subprocess.run(["python", "train.py"], check=True)
    print('==='*30)
    print('\n')

    # Run evaluation script
    print('==='*30)
    print("Starting evaluation...")
    subprocess.run(["python", "evaluate.py"], check=True)
    print('==='*30)

if __name__ == "__main__":
    main()


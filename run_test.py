import os
import argparse
import torch
from metaquantus.configs import (
    setup_estimators,
    setup_xai_methods,
    setup_dataset_models,
    setup_analyser_suite,
)

if __name__ == "__main__":

    print(f"Running from path: {os.getcwd()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_assets")
    parser.add_argument("--dataset_name")

    args = parser.parse_args()
    arguments = {"path_assets": args.path_assets, "dataset_name": args.dataset_name}
    print(arguments)

    # Setting device on GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()
    print(torch.version.cuda)

    # Additional info when using cuda.
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), "GB")

    SETTINGS, model = setup_dataset_models(
        dataset_name=arguments["dataset_name"], device=device
    )

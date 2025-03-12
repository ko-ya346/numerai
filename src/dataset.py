import argparse
import os
from numerapi import NumerAPI

def download_file(api, path, url, description=""):
    """
    Downloads a file from a given URL and saves it to the specified path.

    :param api: NumerAPI instance
    :param path: Destination file path
    :param url: URL of the file to download
    :param description: description of the file for logging
    """
    print(f"Downloading {description} to {path}...")
    api.download_dataset(url, dest_path=path)
    print(f"Downloaded {description} to {path}.")

def main(output_path, version):
    """
    Main function to download the required Numerai files.

    :param output_path: Directory to save the downloaded files
    :param version: Data version to download
    :param feature_set: Feature set name to download
    """
    # Initialize NumerAPI
    napi = NumerAPI()

    all_datasets = napi.list_datasets()
    target_version_datasets = [d for d in all_datasets if version in d]
    print(f"Dataset: {len(target_version_datasets)}")

    # exist_ok が False の場合、対象のディレクトリが存在するとFile ExistsError を返す
    version_output_path = os.path.join(output_path, version)
    os.makedirs(version_output_path, exist_ok=True)

    # Download learning data
    for dataset in target_version_datasets:
        path = os.path.join(output_path, dataset)
        download_file(napi, path, dataset, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Numerai datasets.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save downloaded files")
    parser.add_argument("--version", type=str, required=True, help="Data version to download")
    
    args = parser.parse_args()

    main(args.output_path, args.version)

import os
import json
import matplotlib.pyplot as plt
from pathlib import Path


def find_json_file(folder_path):
    """Find the first JSON file in the given folder."""
    for file in folder_path.iterdir():
        if file.suffix.lower() == '.json':
            return file
    return None


def load_and_process_data(json_path):
    """Load and process data from a JSON file, returning average R² values."""
    try:
        with open(json_path, "r") as file:
            data = json.load(file)

        r2_values = []
        for bucket in range(1, 11):
            bucket_key = f"{bucket}_buckets"
            if bucket_key in data:
                avg_r2 = sum(metric["R2"] for metric in data[bucket_key]) / len(data[bucket_key])
                r2_values.append(avg_r2)
            else:
                r2_values.append(None)

        return r2_values
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return None


def create_comparison_plot(old_values, new_values, subfolder_name, output_path):
    """Create and save a comparison plot of R² values."""
    plt.figure(figsize=(12, 6))

    plt.plot(range(1, 11), old_values, marker='o', label='XGB Old', color='blue')
    plt.plot(range(1, 11), new_values, marker='s', label='XGB New', color='red')

    plt.title(f"R² Comparison for {subfolder_name}")
    plt.xlabel("Number of Buckets")
    plt.ylabel("Average R²")
    plt.ylim(0, 1)
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path("examples/results")
    xgb_old_dir = base_dir / "xgb_old/xgb_old"
    xgb_new_dir = base_dir / "xgboost/one_class"
    # xgb_new_dir = r"C:\Users\karen\Downloads\EIRegression\examples\results\xgboost\one_class"
    # xgb_old_dir = r"C:\Users\karen\Downloads\EIRegression\examples\results\xgb_old\xgb_old"

    # Get all subfolder names from both directories
    old_subfolders = {f.name for f in xgb_old_dir.iterdir() if f.is_dir()}
    new_subfolders = {f.name for f in xgb_new_dir.iterdir() if f.is_dir()}

    # Process only subfolders that exist in both directories
    common_subfolders = old_subfolders.intersection(new_subfolders)
    print(f"Processing subfolders: {common_subfolders}")

    for subfolder_name in common_subfolders:
        print(f"\nProcessing {subfolder_name}...")

        # Get paths for both old and new folders
        old_folder = xgb_old_dir / subfolder_name
        new_folder = xgb_new_dir / subfolder_name

        # Find JSON files
        old_json = find_json_file(old_folder)
        new_json = find_json_file(new_folder)

        if old_json and new_json:
            print(f"Found JSON files:")
            print(f"Old: {old_json}")
            print(f"New: {new_json}")

            # Load data from both files
            old_values = load_and_process_data(old_json)
            new_values = load_and_process_data(new_json)

            if old_values and new_values:
                # Create comparison plot
                plot_filename = base_dir / f"{subfolder_name}_comparison.png"
                create_comparison_plot(old_values, new_values, subfolder_name, plot_filename)
                print(f"Plot saved: {plot_filename}")
            else:
                print(f"Error: Could not process data for {subfolder_name}")
        else:
            print(f"Error: Could not find JSON files in one or both folders")


if __name__ == "__main__":
    main()
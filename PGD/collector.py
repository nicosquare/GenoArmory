import os
import re
import csv
import argparse
from pathlib import Path

def extract_accuracy(file_path):
    """Extract accuracy value from a result.txt file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Look for patterns like "accuracy: 0.95" or "accuracy = 0.95"
            match = re.search(r'accuracy[:\s=]+([\d.]+)', content, re.IGNORECASE)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def collect_accuracies(root_folder):
    """Collect accuracy values from all result.txt files in the folder structure."""
    results = []
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower() in ['result.txt', 'results.txt']:  # Check for both singular and plural
                file_path = os.path.join(root, file)
                accuracy = extract_accuracy(file_path)
                if accuracy is not None:
                    # Get the relative path from the root folder
                    rel_path = os.path.relpath(root, root_folder)
                    results.append({
                        'folder': rel_path,
                        'accuracy': accuracy
                    })
    
    return results

def save_to_csv(results, output_path):
    """Save the results to a CSV file."""
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['folder', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    folder_path = input("Enter the folder path: ")

    # Validate folder path
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return

    # Collect accuracy values
    results = collect_accuracies(folder_path)
    
    if not results:
        print("No accuracy values found in any result.txt files")
        return

    # Save to CSV
    output_path = os.path.join(folder_path, 'accuracy_results.csv')
    save_to_csv(results, output_path)
    print(f"Results saved to {output_path}")
    print(f"Found {len(results)} accuracy values")

if __name__ == "__main__":
    main() 
import os
import json
import pandas as pd
from pathlib import Path

def collect_metrics(folder_path):
    # Initialize lists to store metrics and paths
    metrics1_accs = []
    metrics_accs = []
    json_paths = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist!")
        return
    
    print(f"\nSearching in folder: {folder_path}")
    
    # Get all immediate subfolders
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    if not subfolders:
        print("❌ No subfolders found in the given path!")
        return
    
    print(f"Found {len(subfolders)} subfolders")
    
    # Process each subfolder
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, 'all_results.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'metrics1' in data and 'metrics' in data:
                        metrics1_accs.append(data['metrics1']['accuracy'])
                        metrics_accs.append(data['metrics']['accuracy'])
                        json_paths.append(json_path)
                        print(f"✓ Found metrics in: {json_path}")
                    else:
                        print(f"⚠ Found all_results.json but missing required metrics in: {json_path}")
            except json.JSONDecodeError:
                print(f"⚠ Error: Invalid JSON format in {json_path}")
            except Exception as e:
                print(f"⚠ Error reading {json_path}: {str(e)}")
        else:
            print(f"⚠ No all_results.json found in: {subfolder}")
    
    if not metrics1_accs:
        print("\n❌ No valid metrics found in any subfolders!")
        print("Please check that:")
        print("1. The folder path is correct")
        print("2. all_results.json files exist in subdirectories")
        print("3. The JSON files contain 'metrics1' and 'metrics' fields with accuracy values")
        return
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'org_accuracy': metrics1_accs,
        'adv_accuracy': metrics_accs,
        'json_path': json_paths
    })
    
    # Calculate ASR
    df['asr'] = (df['org_accuracy'] - df['adv_accuracy']) / df['org_accuracy']
    
    # Save to CSV in the given folder
    output_path = os.path.join(folder_path, 'collected_metrics.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully collected {len(metrics1_accs)} sets of metrics")
    print(f"Metrics saved to: {output_path}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    collect_metrics(folder_path) 
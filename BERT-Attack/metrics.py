import json
import pandas as pd
import os
from pathlib import Path

# Function to calculate metrics
def calculate_metrics(data):
    """
    Calculate metrics from the input DataFrame:
    - Success rate for records where success = 4.
    - Average change rate based on change/num_word.
    - Origin accuracy (origin_acc): success3 / total_records.
    - After attack accuracy (aft_atk_acc): (success1 + success2) / total_records.

    Args:
        data (pd.DataFrame): DataFrame with columns 'success', 'change', and 'num_word'.

    Returns:
        tuple: (success_rate_4, average_change_rate, origin_acc, aft_atk_acc)
    """
    # Total number of records
    total_records = len(data)

    # Calculate success rate for success = 4
    success_count_4 = (data["success"] == 4).sum()
    success_rate_4 = success_count_4 / total_records if total_records > 0 else 0

    # Calculate average change rate
    data["change_rate"] = data["change"] / data["num_word"]
    average_change_rate = data["change_rate"].mean() if total_records > 0 else 0

    # Calculate origin_acc (success3 / total_records)
    success_count_3 = (data["success"] == 3).sum()
    origin_acc = success_count_3 / total_records if total_records > 0 else 0
    origin_acc = 1 - origin_acc

    # Calculate aft_atk_acc ((success1 + success2) / total_records)
    success_count_1_and_2 = (data["success"] == 1).sum() + (data["success"] == 2).sum()
    aft_atk_acc = success_count_1_and_2 / total_records if total_records > 0 else 0
    return success_rate_4, average_change_rate, origin_acc, aft_atk_acc

def process_directory(directory_path):
    """
    Process all JSON files in a directory and return results as a DataFrame.
    
    Args:
        directory_path (str): Path to directory containing JSON files
        
    Returns:
        pd.DataFrame: DataFrame with filename and aft_atk_acc columns
    """
    results = []
    directory = Path(directory_path)
    
    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = pd.read_json(f)
            _, _, origin_acc, aft_atk_acc = calculate_metrics(data)
            results.append({
                'filename': json_file.name,
                'aft_atk_acc': aft_atk_acc,
                'origin_acc': origin_acc,
                'ASR': (origin_acc - aft_atk_acc) / origin_acc if origin_acc > 0 else 0,
            })
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Get directory path from user
    directory_path = input("Enter the directory path containing JSON files: ")
    
    # Process all JSON files and get results
    results_df = process_directory(directory_path)
    
    # Sort by filename for consistent display
    results_df = results_df.sort_values('filename')
    
    # Display results in a nice format
    print("\nResults:")
    print("=" * 50)
    print(results_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))
    print("=" * 50)
    
    # Save results to CSV
    output_dir = directory_path
    output_file = output_dir + "/attack_results.csv"
    results_df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\nResults saved to: {output_file}")

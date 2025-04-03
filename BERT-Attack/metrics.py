import json
import pandas as pd
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


# Example input
input_json = input("Enter your JSON data: ")
data = pd.read_json(input_json)

# # Calculate and display the metrics
success_rate, avg_change_rate, origin_acc, aft_atk_acc = calculate_metrics(data)
print(f"Success Rate (success = 4): {success_rate:.4f}")
print(f"Average Change Rate: {avg_change_rate:.4f}")
print(f"origin_acc: {origin_acc:.4f}")
print(f"aft_atk_acc: {aft_atk_acc:.4f}")

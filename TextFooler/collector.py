import os
import re
import pandas as pd

def extract_accuracies(log_line):
    # Extract original and adversarial accuracies using regex
    orig_acc_match = re.search(r'original accuracy: (\d+\.\d+)%', log_line)
    adv_acc_match = re.search(r'adv accuracy: (\d+\.\d+)%', log_line)
    
    if orig_acc_match and adv_acc_match:
        return float(orig_acc_match.group(1)), float(adv_acc_match.group(1))
    return None, None

def process_results():
    base_path = input("Enter the base path: ")
    results = []
    
    # Walk through all folders in the base path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            log_file = os.path.join(folder_path, 'results_log')
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    first_line = f.readline().strip()
                    orig_acc, adv_acc = extract_accuracies(first_line)
                    
                    if orig_acc is not None and adv_acc is not None:
                        # Calculate ASR (Attack Success Rate)
                        if orig_acc!=0:
                            asr = (orig_acc - adv_acc) / orig_acc
                        else: 
                            asr = 0

                        results.append({
                            'folder_name': folder_name,
                            'original_accuracy': orig_acc,
                            'adversarial_accuracy': adv_acc,
                            'ASR': asr
                        })
                        
    
    # Create DataFrame and save to CSV in base_path
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(base_path, 'accuracy_results.csv')
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results found")

if __name__ == "__main__":
    process_results() 
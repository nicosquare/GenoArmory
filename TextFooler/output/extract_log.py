import os
import csv

root_dir = "og"

csv_output_path = "og_summary_results_new.csv"

header = ["Task Name", "BERTAttack", "TextFooler"]
rows = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "results_log" in filenames:
        task_name = os.path.basename(dirpath)
        log_path = os.path.join(dirpath, "results_log")
        with open(log_path, "r") as f:
            for line in f:
                if ":" in line:
                    # 只取第一个冒号之后的内容
                    content_after_colon = line.split(":", 1)[1].strip()
                    rows.append([task_name, "", content_after_colon])
                    break  # 只取第一行

# 按任务名排序
rows.sort(key=lambda x: x[0])

# 写入 CSV 文件
with open(csv_output_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

print(f"✅ CSV 文件已生成：{csv_output_path}")

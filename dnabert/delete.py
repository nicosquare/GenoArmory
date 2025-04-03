import os
import shutil

def delete_checkpoint_folders(directory):
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在！")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            print(f"正在删除: {item_path}")
            shutil.rmtree(item_path)
    
    print("删除完成！")

if __name__ == "__main__":
    target_directory = "/projects/p32013/DNABERT-meta/dnabert/output_pipe"
    delete_checkpoint_folders(target_directory)

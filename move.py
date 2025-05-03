import os
import shutil
import sys

def move_checkpoint_contents(base_path):
    """
    Recursively traverse through directories and move contents from checkpoint folders
    to their parent directories.
    
    Args:
        base_path (str): The root directory path to start searching from
    """
    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist")
        return

    # Walk through all directories
    for root, dirs, files in os.walk(base_path):
        # Look for directories that start with 'checkpoint'
        checkpoint_dirs = [d for d in dirs if d.startswith('checkpoint')]
        
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = os.path.join(root, checkpoint_dir)

            
            # Get all contents from the checkpoint directory
            for item in os.listdir(checkpoint_path):
                src = os.path.join(checkpoint_path, item)
                dst = os.path.join(root, item)
                try:
                    # If it's a file, copy it
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        print(f"Copied file: {src} -> {dst}")
                    # If it's a directory, copy the entire directory
                    elif os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        print(f"Copied directory: {src} -> {dst}")
                except Exception as e:
                    print(f"Error copying {src}: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python move.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    print(f"Processing directory: {directory_path}")
    move_checkpoint_contents(directory_path)
    print("Processing complete!")

if __name__ == "__main__":
    main()

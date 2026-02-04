import os

# Configuration
OUTPUT_FILE = "mega_project_dump.txt"
# Directories to ignore
IGNORE_DIRS = {'.git', '__pycache__', 'venv', 'env', '.idea', '.vscode', 'node_modules'}
# File extensions to ignore (binary files, etc.)
IGNORE_EXTENSIONS = {'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.exe', '.bin'}

DELIMITER_START = "====== FILE_START: "
DELIMITER_END = "====== FILE_END ======"

def is_text_file(file_path):
    """Check if file is text by trying to read the first block."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, IOError):
        return False

def pack_project():
    root_dir = os.path.abspath(os.getcwd())
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        print(f"Scanning directory: {root_dir}")
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Modify dirnames in-place to skip ignored directories
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            
            for filename in filenames:
                # Skip the output file itself to prevent infinite recursion
                if filename == OUTPUT_FILE:
                    continue
                
                # Check extension
                if any(filename.endswith(ext) for ext in IGNORE_EXTENSIONS):
                    continue
                
                file_path = os.path.join(dirpath, filename)
                
                # Verify it's a text file before appending
                if is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                        # Write the start delimiter with absolute path
                        outfile.write(f"{DELIMITER_START}{file_path}\n")
                        outfile.write(content)
                        # Ensure content ends with a newline before delimiter
                        if content and not content.endswith('\n'):
                            outfile.write('\n')
                        outfile.write(f"{DELIMITER_END}\n")
                        
                        print(f"Packed: {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                else:
                    print(f"Skipping binary file: {filename}")

    print(f"\n--- Success! All files packed into {OUTPUT_FILE} ---")

if __name__ == "__main__":
    pack_project()

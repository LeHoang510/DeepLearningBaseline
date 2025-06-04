import os
import shutil

# Đường dẫn chính xác đến các thư mục cần làm sạch (tương đối hoặc tuyệt đối)
FOLDERS_TO_CLEAN = [
    "logs",
    "outputs"
]

def clean_specified_folders(root_dir="."):
    deleted_files = 0
    deleted_dirs = 0

    for folder in FOLDERS_TO_CLEAN:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.exists(folder_path):
            continue

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    print(f"🗑️  Deleted file: {item_path}")
                    deleted_files += 1
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"🧹 Deleted folder: {item_path}")
                    deleted_dirs += 1
            except Exception as e:
                print(f"❌ Error deleting {item_path}: {e}")

    print(f"\n✅ Done. Deleted {deleted_files} files and {deleted_dirs} folders inside specified folders.")

if __name__ == "__main__":
    clean_specified_folders()

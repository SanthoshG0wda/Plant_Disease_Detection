import os
import shutil
import random

SOURCE_DIR = "model_1_data"          # your current dataset
DEST_DIR = "model_1_dataset"      # new folder to be created

TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO  = 0.1

# Create base directories
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    splits = {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:]
    }

    for split, split_images in splits.items():
        split_class_dir = os.path.join(DEST_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)

    print(f"✔ Split completed for class: {class_name}")

print("\n✅ Dataset successfully split into train / valid / test")

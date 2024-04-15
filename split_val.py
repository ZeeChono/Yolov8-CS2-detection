# This script randomly split validation data
import os
import random
import shutil


if __name__ == '__main__':
    # Source and destination directories
    label_source_dir = './data/labels/train'
    label_destination_dir = './data/labels/val'
    img_source_dir = './data/images/train'
    img_destination_dir = './data/images/val'

    # Ensure destination directory exists
    if not os.path.exists(label_destination_dir):
        os.makedirs(label_destination_dir)
    if not os.path.exists(img_destination_dir):
        os.makedirs(img_destination_dir)

    # List all .txt files in the source directory
    file_names = set(os.path.splitext(file)[0] for file in os.listdir(label_source_dir))

    # Randomly select files
    random.seed(42)
    num_of_val = 64
    selected_files = random.sample(file_names, num_of_val)

    for file_name in selected_files:
        source_label = os.path.join(label_source_dir, file_name+".txt")
        destination_label = os.path.join(label_destination_dir, file_name+".txt")
        source_img = os.path.join(img_source_dir, file_name+".jpg")
        destination_img = os.path.join(img_destination_dir, file_name+".jpg")

        shutil.move(source_label, destination_label)
        shutil.move(source_img, destination_img)
        # print(f"Moved {file_name} to {destination_label}")
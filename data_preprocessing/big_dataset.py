import os
import shutil
import random


src_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\cropped_image"
dst_dir = r"F:\FAU\Thesis\HDMIBabelfishV2\data\game_review\big_dataset"
train_dir = os.path.join(dst_dir, "train")
test_dir = os.path.join(dst_dir, "test")


os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ----- Step 1: Gather Labeled Images -----
labeled_files = []
for file in os.listdir(src_dir):
    if file.lower().endswith('.jpg'):
        base_name = os.path.splitext(file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(src_dir, label_file)
        if os.path.isfile(label_path):  # Only if label exists, consider it labeled.
            labeled_files.append(file)

total_labeled = len(labeled_files)
print(f"Total labeled images found: {total_labeled}")

# ----- Step 2: Split into Train/Test -----
if total_labeled >= 1000:
    num_train = 800
    num_test = 200
else:
    num_train = int(total_labeled * 0.8)
    num_test = total_labeled - num_train


random.shuffle(labeled_files)
train_files = labeled_files[:num_train]
test_files = labeled_files[num_train:num_train+num_test]

print(f"Selected {len(train_files)} images for training and {len(test_files)} images for testing.")

# ----- Step 3: Copy Files to Destination -----
def copy_files(file_list, target_dir):
    for file in file_list:
        base_name = os.path.splitext(file)[0]
        image_src = os.path.join(src_dir, file)
        label_src = os.path.join(src_dir, base_name + '.txt')

        image_dst = os.path.join(target_dir, file)
        label_dst = os.path.join(target_dir, base_name + '.txt')


        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)


print("Copying training files...")
copy_files(train_files, train_dir)


print("Copying testing files...")
copy_files(test_files, test_dir)

# ----- Step 4: Copy classes.txt -----
classes_file_src = os.path.join(src_dir, "classes.txt")
classes_file_dst = os.path.join(dst_dir, "classes.txt")

if os.path.isfile(classes_file_src):
    shutil.copy2(classes_file_src, classes_file_dst)
    print("classes.txt copied to destination.")
else:
    print("Warning: classes.txt not found in source folder.")

print("Dataset creation complete!")

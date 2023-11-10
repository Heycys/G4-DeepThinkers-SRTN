import os
import shutil

# Source folder path
src_dir = "E:\\breast\\BreaKHis_v1\\histology_slides\\breast"

# Destination folder path
dest_dir = "E:\\breast\\100X"

# Create the destination folder if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Traverse all files and folders in the source folder
for root, dirs, files in os.walk(src_dir):
    # Check if the current directory is a "100X" folder
    if '100X' in root:
        for file in files:
            # Check if the file is an image (assuming images are in png or jpg format)
            if file.endswith(('.png', '.jpg', '.jpeg')):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                # Check if a file with the same name already exists in the destination directory
                if os.path.exists(dest_file):
                    # If it exists, modify the file name to avoid overwriting
                    basename, extension = os.path.splitext(file)
                    dest_file = os.path.join(dest_dir, basename + "_duplicate" + extension)

                # Copy the file
                shutil.copy2(src_file, dest_file)

print("All '100X' images have been copied to " + dest_dir)

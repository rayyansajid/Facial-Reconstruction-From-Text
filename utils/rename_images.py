import os

def rename_images(folder_path, start_number=202600):
    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter files with valid image extensions
    images = [file for file in files if os.path.splitext(file)[1].lower() in valid_extensions]

    # Sort images alphabetically (optional, for consistent order)
    images.sort()

    # Rename each image
    for index, image in enumerate(images):
        # Generate new name
        new_name = f"{start_number + index}.jpg"

        # Get full paths
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, new_name)

        # Rename file
        os.rename(old_path, new_path)
        print(f"Renamed: {image} -> {new_name}")

# Set the path to your folder containing images
folder_path = r"e:\FYDP Dataset\fyp_dataset\images"  # Replace this with the actual folder path

# Call the function to rename images
rename_images(folder_path)

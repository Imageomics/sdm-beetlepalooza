### Segmented images for background removal can be retrieved from https://huggingface.co/datasets/imageomics/2018-NEON-beetles/tree/main/separate_segmented_beetle_images ###

import cv2
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# Define the image processing function
def process_image(image):
    # Step 1: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Otsu's thresholding to create a binary mask
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Step 3: Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    # Step 4: Apply morphological operations to clean up the mask
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Step 5: Convert the cleaned mask to three channels
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Step 6: Create a white background image
    white_background = np.ones_like(image) * 255
    
    # Step 7: Use the mask to remove the background
    foreground = cv2.bitwise_and(image, mask_3ch)
    background_removed = cv2.bitwise_or(foreground, cv2.bitwise_and(white_background, cv2.bitwise_not(mask_3ch)))

    return background_removed, gray, binary, mask

# Set the parent directory containing all subfolders with images
parent_dir = ''  # Replace with the path to your parent folder
output_dir = ''  # Replace with the path to save processed images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all subdirectories and files in the parent directory
for root, dirs, files in os.walk(parent_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):  # Filter image files
            # Get the full file path
            image_file = os.path.join(root, file)
            
            # Load the image
            image = cv2.imread(image_file)
            
            if image is None:
                print(f"Skipping {image_file}: Unable to read image.")
                continue

            # Process the image
            background_removed, gray, binary, mask = process_image(image)
            
            # Convert the processed image from BGR to RGB for saving and displaying
            background_removed_rgb = cv2.cvtColor(background_removed, cv2.COLOR_BGR2RGB)
            
            # Create a PIL image from the processed data
            result_pil = Image.fromarray(background_removed_rgb)
            
            # Determine the output path, keeping the subfolder structure
            relative_path = os.path.relpath(root, parent_dir)
            save_dir = os.path.join(output_dir, relative_path)
            os.makedirs(save_dir, exist_ok=True)  # Ensure the subdirectory exists
            
            # Save the processed image in the corresponding output directory
            result_pil.save(os.path.join(save_dir, file))
            
            print(f"Processed and saved: {os.path.join(save_dir, file)}")

            # Display the original and processed images for review
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(2, 3, 2)

import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get the project's root directory
project_root = os.path.abspath(os.path.dirname(__file__))

# Define paths to your dataset folders
conjunctivitis_dir = os.path.join(project_root, "Dataset", "Conjunctivitis")
normal_eye_dir = os.path.join(project_root, "Dataset", "Non Conjunctivities")
output_dir = os.path.join(project_root, "output")  # Where augmented images will be saved

# Create an instance of the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the number of augmentations per image
augmentation_factor = 5  # Adjust as needed

# Function to augment and save images
def augment_and_save_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all image files in the input directory
    image_files = os.listdir(input_dir)

    for image_file in image_files:
        # Load the image using OpenCV
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)

        # Check if the image is successfully loaded
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Expand the image dimensions to (1, height, width, channels) for processing
        img = img.reshape((1,) + img.shape)

        # Generate augmented images
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_image = batch[0]

            # Save the augmented image to the output directory
            output_image_path = os.path.join(output_dir, f"aug_{i}_{image_file}")
            cv2.imwrite(output_image_path, augmented_image)

            i += 1
            if i >= augmentation_factor:
                break

# Augment conjunctivitis images
augment_and_save_images(conjunctivitis_dir, os.path.join(output_dir, "conjunctivitis"))

# Augment normal eye images
augment_and_save_images(normal_eye_dir, os.path.join(output_dir, "normal"))

print("Data augmentation complete.")

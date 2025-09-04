from PIL import Image
import os

def validate_images(folder):
    for subdir, _, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # verify() doesn't load the full image, just header
            except Exception as e:
                print(f"Deleting invalid image: {filepath} ({e})")
                os.remove(filepath)

# Use this on both train and test directories
validate_images(r"C:\Users\HI\OneDrive\Desktop\Brain_Tumor_Detection_and_Classification\dataset\Training")
validate_images(r"C:\Users\HI\OneDrive\Desktop\Brain_Tumor_Detection_and_Classification\dataset\Testing")

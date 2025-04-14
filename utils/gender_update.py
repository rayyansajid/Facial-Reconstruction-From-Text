import pandas as pd
import os
import re
import logging
from deepface import DeepFace
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CSV_PATH = os.getenv("CORRECTION_CSV")
IMAGE_FOLDER_PATH = os.getenv("IMAGE_FOLDER_PATH")

# Configure logging
logging.basicConfig(
    filename="./logs/gender_description_correction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting gender description correction process.")
logging.info(f"CSV Path: {CSV_PATH}")
logging.info(f"Image Folder Path: {IMAGE_FOLDER_PATH}")

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
    logging.info("CSV loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load CSV: {e}")
    raise

# Drop rows with missing values in essential columns
df.dropna(subset=["Image Name", "Description"], inplace=True)

# Function to check if gender is already mentioned
def has_gender(description):
    gender_keywords = ["male", "female", "man", "woman", "boy", "girl", "he", "she", "his", "her"]
    return any(word.lower() in gender_keywords for word in description.split())

# Function to replace placeholders with gendered terms
def gender_replace(description, gender):
    if gender == "Male":
        replacements = {
            r'\b[Pp]erson\b': 'male',
            r'\b[Tt]hey\b': 'He',
            r'\b[Tt]heir\b': 'His',
            r'\b[Tt]hem\b': 'Him'
        }
    else:
        replacements = {
            r'\b[Pp]erson\b': 'female',
            r'\b[Tt]hey\b': 'She',
            r'\b[Tt]heir\b': 'Her',
            r'\b[Tt]hem\b': 'Her'
        }
    for pattern, replacement in replacements.items():
        description = re.sub(pattern, replacement, description)
    return description

# Predict gender using DeepFace
def predict_gender(image_path):
    try:
        objs = DeepFace.analyze(
            img_path=rf"{image_path}",
            actions=["gender"],
            enforce_detection=False,
        )
        dominant_gender = objs[0]['dominant_gender']
        logging.info(f"Predicted gender for {image_path}: {dominant_gender}")

        with open("./correct_genders.txt", "a+") as f:
            f.write(f"{os.path.basename(image_path)},{dominant_gender}\n")

        # Normalize to "Male" / "Female"
        if dominant_gender.lower() in ['woman', 'female']:
            return "Female"
        else:
            return "Male"

    except Exception as e:
        logging.error(f"Gender prediction failed for {image_path}: {e}")
        return "Male"  # Default fallback if detection fails

# Function to update description based on predicted gender
def update_description(image_path, description):
    if has_gender(description):
        logging.info(f"Gender already mentioned in description for {image_path}.")
        return description

    gender = predict_gender(image_path)
    updated_description = gender_replace(description, gender)
    logging.info(f"Updated description for {image_path}.")
    return updated_description

# Apply updates
df["updated_description"] = df.apply(
    lambda row: update_description(f"{IMAGE_FOLDER_PATH}/{row['Image Name']}", row["Description"]),
    axis=1
)

# Save updated CSV (image name and updated description only)
output_path = "E:/FYDP Dataset/Corrections/subset_with_gender.csv"
try:
    df[["Image Name", "updated_description"]].to_csv(output_path, index=False, header=False)
    logging.info(f"Updated CSV saved to {output_path}")
except Exception as e:
    logging.error(f"Failed to save updated CSV: {e}")

logging.info("Gender description correction process completed.")

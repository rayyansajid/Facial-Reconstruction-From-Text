import os
import torch
import csv
import sys
import logging
from datetime import datetime
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, LlamaForCausalLM, LlamaTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from my_mp_attr import extract_mediapipe_attributes
from deepface import DeepFace
from dotenv import load_dotenv

# Setup logging
log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(    
    filename=f"./logs/{log_filename}",
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

IMAGE_FOLDER_PATH = os.getenv("IMAGE_FOLDER_PATH")
CSV_PATH = os.getenv("CSV_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
API_PATH = os.getenv("API_PATH")

attribute_names = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
    "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
    "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face",
    "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling",
    "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

def load_model(model_path, device):
    logging.info("Loading model from %s", model_path)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 40)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info("Model loaded successfully")
    return model

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0)
        input_image = input_image.to(device)
        return input_image
    except (FileNotFoundError, OSError) as e:
        raise ValueError(f"Error loading image at {image_path}: {e}")

def make_predictions(model, input_image):
    with torch.no_grad():
        outputs = model(input_image)
    probabilities = torch.sigmoid(outputs).cpu().numpy()
    predictions = (probabilities > 0.5).astype(int)
    return predictions

def interpret_predictions(predictions, mediapipe_attributes, api_keys):
    predicted_attributes = [attribute_names[i] for i, val in enumerate(predictions[0]) if val == 1]
    combined_attributes = predicted_attributes + [f"{k}: {v}" for k, v in mediapipe_attributes.items()]

    for api_key in api_keys:
        try:
            chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
            system = "You are a helpful assistant and does not generate anything else other than face description"
            human = "Generate a natural language description of the face appearance from {text}. include all given attributes in a natural language description. Don't exaggerate the gender, just mention it once."
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

            chain = prompt | chat
            res = chain.invoke({"text": f"These are facial attributes detected from an image: {combined_attributes}"})
            return res.content
        except Exception as e:
            logging.warning("API key %s failed with error: %s", api_key, e)
            continue

    raise RuntimeError("All API keys exhausted or token limits reached.")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

folder_path = rf"{IMAGE_FOLDER_PATH}"
model_path = rf"{MODEL_PATH}"
output_csv = rf"{CSV_PATH}"

# Load API keys
api_keys = []
try:
    with open(API_PATH, mode="r", encoding="utf-8") as file:
        api_keys = [key.strip() for key in file.readlines()]
except Exception as e:
    logging.error("Failed to read API keys: %s", e)
    sys.exit(1)

if not api_keys:
    logging.error("No API keys found. Exiting.")
    sys.exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using device: %s", device)

# Load model
model = load_model(model_path, device)

try:
    start_index = int(input("Enter the starting index: "))
except ValueError:
    logging.error("Invalid starting index. Please provide a valid integer.")
    sys.exit(1)

file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0

with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Image Name", "Description"])

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for index, image_name in enumerate(image_files, start=1):
        if index < start_index:
            continue

        image_path = os.path.join(folder_path, image_name)
        logging.info("Processing %s (%d/%d)", image_name, index, len(image_files))

        try:
            input_image = preprocess_image(image_path, device)
            predictions = make_predictions(model, input_image)
            mediapipe_attributes = extract_mediapipe_attributes(image_path)
            description = interpret_predictions(predictions, mediapipe_attributes, api_keys)

            logging.info("Generated description: %s", description)

            writer.writerow([image_name, description])
            logging.info("Saved description for %s", image_name)

        except ValueError as e:
            logging.warning("Skipping %s due to error: %s", image_name, e)
        except RuntimeError as e:
            logging.error("Stopping processing due to error: %s", e)
            break

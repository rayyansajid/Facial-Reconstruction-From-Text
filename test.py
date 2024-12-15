import os
import torch
import csv
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, LlamaForCausalLM, LlamaTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from my_mp_attr import extract_mediapipe_attributes
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()
IMAGE_FOLDER_PATH=os.getenv("IMAGE_FOLDER_PATH")
CSV_PATH=os.getenv("CSV_PATH")
IMAGE_PATH = os.getenv("IMAGE_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")   
API_PATH=os.getenv("API_PATH") 

# List of attribute names from CelebA dataset
attribute_names = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", 
    "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", 
    "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", 
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", 
    "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", 
    "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", 
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

# Function to load the trained model
def load_model(model_path, device):
    model = models.resnet18(pretrained=False)  # Set pretrained to False for inference
    model.fc = nn.Linear(model.fc.in_features, 40)  # Ensure output size matches training
    model = model.to(device)
    
    model.load_state_dict(torch.load(model_path))  # Load trained weights
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess the image
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image = Image.open(image_path).convert("RGB")  # Open and convert image
        input_image = transform(image).unsqueeze(0)  # Add batch dimension
    except (FileNotFoundError, OSError) as e:
        raise ValueError(f"Error loading image at {image_path}: {e}")

    input_image = input_image.to(device)  # Move image to the device (GPU/CPU)
    return input_image

# Function to make predictions
def make_predictions(model, input_image):
    with torch.no_grad():  # No gradients needed for inference
        outputs = model(input_image)

    probabilities = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities
    predictions = (probabilities > 0.5).astype(int)  # Convert to binary predictions (threshold = 0.5)
    return predictions

# Function to interpret predictions and generate description
def interpret_predictions(predictions, mediapipe_attributes, api_keys):
    predicted_attributes = [attribute_names[i] for i, val in enumerate(predictions[0]) if val == 1]
    combined_attributes = predicted_attributes + [f"{k}: {v}" for k, v in mediapipe_attributes.items()]

    for api_key in api_keys:
        try:
            chat = ChatGroq(temperature=0,
                            groq_api_key=api_key, 
                            model_name="llama-3.1-70b-versatile")

            system = "You are a helpful assistant and does not generate anything else other than face description"
            human = "Generate a natural language description of the face appearance from {text}. include all given attributes in a natural language description. Don't exaggerate the gender. "
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

            chain = prompt | chat
            res = chain.invoke({"text": f"These are facial attributes detected from an image: {combined_attributes}"})
            return res.content
        except Exception as e:
            print(f"API key {api_key} failed with error: {e}")
            continue

    raise RuntimeError("All API keys exhausted or token limits reached.")

# Main function
def main():
    # Suppress TensorFlow warnings
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    folder_path = rf"{IMAGE_FOLDER_PATH}"  # Folder containing images
    model_path = rf"{MODEL_PATH}"  # Path to your saved model
    output_csv = rf"{CSV_PATH}"  # Path to save the CSV file

    # Load API keys from the txt file
    api_keys = []
    with open(API_PATH, mode="r", encoding="utf-8") as file:
        keys = file.readlines()
        for key in keys:
            api_keys.append(key.strip('\n'))
    print(f"@@@@@@@@@api_keys: {api_keys}")
    if not api_keys:
        print("No API keys found. Exiting.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device (GPU/CPU)
    print(f"Using device: {device}")

    # Load the model
    model = load_model(model_path, device)

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Description"])

        # Process each image in the folder
        for image_name in sorted(os.listdir(folder_path)):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, image_name)
                print(f"Processing {image_name}...")

                try:
                    # Preprocess the image
                    input_image = preprocess_image(image_path, device)

                    # Make predictions
                    predictions = make_predictions(model, input_image)

                    # Extract mediapipe attributes
                    mediapipe_attributes = extract_mediapipe_attributes(image_path)

                    # Generate description
                    description = interpret_predictions(predictions, mediapipe_attributes, api_keys)

                    # Write to CSV
                    writer.writerow([image_name, description])
                    print(f"Processed {image_name}: {description}")
                except ValueError as e:
                    print(f"Skipping {image_name} due to error: {e}")
                except RuntimeError as e:
                    print(f"Stopping processing due to error: {e}")
                    break

if __name__ == '__main__':
    main()

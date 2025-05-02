import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from my_mp_attr import extract_mediapipe_attributes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")

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
    image = Image.open(image_path).convert("RGB")  # Open and convert image
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    input_image = input_image.to(device)  # Move image to the device (GPU/CPU)
    return input_image

# Function to make predictions
def make_predictions(model, input_image):
    with torch.no_grad():  # No gradients needed for inference
        outputs = model(input_image)
    probabilities = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities
    predictions = (probabilities > 0.5).astype(int)  # Convert to binary predictions (threshold = 0.5)
    return predictions

# Function to generate natural language description
def generate_description(predicted_attributes, mediapipe_attributes):
    combined_attributes = predicted_attributes + [f"{k}: {v}" for k, v in mediapipe_attributes.items()]
    chat = ChatGroq(temperature=0,
                    groq_api_key=API_KEY,
                    model_name="llama-3.3-70b-versatile")
    system = "You are a helpful assistant and does not generate anything else other than face description"
    human = "Generate a natural language description of the face appearance from {text}. Include all given attributes in a natural language description. Don't exaggerate the gender. Just mention it once"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    res = chain.invoke({"text": f"These are facial attributes detected from an image: {combined_attributes}"})
    return res.content

# Streamlit app
def main():
    st.title("Image Description Generator")
    st.write("Upload an image to generate a natural language description of the detected attributes.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Save the uploaded file locally
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        image = Image.open("uploaded_image.jpg")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Set device (GPU/CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        st.write(f"Using device: {device}")

        # Load the model
        model = load_model(MODEL_PATH, device)

        # Preprocess the image
        input_image = preprocess_image("uploaded_image.jpg", device)

        # Make predictions
        predictions = make_predictions(model, input_image)

        # Interpret predictions
        mediapipe_attributes = extract_mediapipe_attributes("uploaded_image.jpg")
        predicted_attributes = [attribute_names[i] for i, val in enumerate(predictions[0]) if val == 1]
        st.write("Detected Attributes:", predicted_attributes)

        # Generate description
        description = generate_description(predicted_attributes, mediapipe_attributes)
        st.write("Generated Description:")
        st.success(description)

if __name__ == "__main__":
    main()
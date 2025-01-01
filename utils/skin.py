# import os
# import cv2
# import numpy as np
# import csv
# google_api_key = ""
# def predict_skin_tone(image_folder):
#     def calculate_skin_tone(image):
#         """
#         Predicts the skin tone of the face in the given image.
#         Parameters:
#             image (numpy.ndarray): Image loaded in BGR format.
#         Returns:
#             str: Predicted skin tone ("Fair", "Medium", or "Dark").
#         """
#         # Convert image to LAB color space
#         lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
#         # Define coordinates for the cheek region (approximate for a centered face)
#         height, width, _ = image.shape
#         cheek_x = int(width * 0.5)  # Center of the image width
#         cheek_y = int(height * 0.6)  # Slightly below the center of the height
#         cheek_region_size = 20  # Adjust the size of the cheek region
        
#         cheek_region = lab_image[
#             max(0, cheek_y - cheek_region_size):min(height, cheek_y + cheek_region_size),
#             max(0, cheek_x - cheek_region_size):min(width, cheek_x + cheek_region_size)
#         ]
        
#         # If the cheek region is invalid, assign as "Medium"
#         if cheek_region.size == 0:
#             return "Medium"
        
#         # Compute average LAB values
#         avg_l, avg_a, avg_b = np.mean(cheek_region, axis=(0, 1))
        
#         # Normalize LAB values for skin tone classification
#         normalized_l = avg_l / 255
#         normalized_a = (avg_a - 128) / 128
#         normalized_b = (avg_b - 128) / 128

#         # Adjusted thresholds for skin tone classification
#         if normalized_l > 0.65:
#             return "Fair"
#         elif 0.4 <= normalized_l <= 0.65:
#             return "Medium"
#         else:
#             return "Dark"

#     # Check if the folder exists
#     if not os.path.exists(image_folder):
#         print(f"Error: The folder path '{image_folder}' does not exist.")
#         return
    
#     # Get all image files in the folder
#     image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     if not image_files:
#         print("No image files found in the specified folder.")
#         return
    
#     # Process each image
#     with open('skin_tone_predictions.csv', mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Image Name", "Skin Tone"])
#         for image_name in image_files:
#             image_path = os.path.join(image_folder, image_name)
#             print(f"Processing {image_name}...")
            
#             try:
#                 # Load the image
#                 image = cv2.imread(image_path)
#                 if image is None:
#                     print(f"Could not load image: {image_name}")
#                     continue
                
#                 # Predict skin tone
#                 skin_tone = calculate_skin_tone(image)
                
#                 if skin_tone:
#                     writer.writerow([image_name, skin_tone])
#                 else:
#                     print(f"Could not determine skin tone for {image_name}")
            
#             except Exception as e:
#                 print(f"Error processing {image_name}: {e}")

# # Example usage
# # predict_skin_tone(r"C:\Users\Rayyan Sajid\OneDrive\Desktop\FYDP\Dataset\facial-reconstruction-from-text\utils\images")
# # Example usage
# predict_skin_tone(r"e:\FYDP Dataset\Final_Dataset\images")



######################################################

import os
import cv2
import numpy as np
import csv

def calculate_skin_tone(image, face_cascade):
    """
    Predicts the skin tone of the face in the given image using face detection.
    Parameters:
        image (numpy.ndarray): Image loaded in BGR format.
        face_cascade (cv2.CascadeClassifier): Haar Cascade classifier for face detection.
    Returns:
        str: Predicted skin tone ("Fair", "Medium", or "Dark").
    """
    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) > 0:
        # Assume the first detected face is the primary face
        x, y, w, h = faces[0]
        
        # Define cheek region as lower-middle part of the detected face
        cheek_region = image[y + h // 2:y + int(h * 0.75), x + w // 4:x + 3 * w // 4]
    else:
        # Fallback: use the center region of the image if no face is detected
        height, width, _ = image.shape
        center_x, center_y = width // 2, height // 2
        region_size = 40
        cheek_region = image[
            max(0, center_y - region_size):min(height, center_y + region_size),
            max(0, center_x - region_size):min(width, center_x + region_size)
        ]
    
    # Ensure the cheek region is valid; if not, use the entire image
    if cheek_region.size == 0:
        cheek_region = image
    
    # Convert cheek region to LAB color space
    lab_image = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2Lab)
    
    # Compute average LAB values
    avg_l, avg_a, avg_b = np.mean(lab_image, axis=(0, 1))
    
    # Normalize LAB values for skin tone classification
    normalized_l = avg_l / 255

    # Adjusted thresholds for skin tone classification
    if normalized_l > 0.65:
        return "Fair"
    elif 0.4 <= normalized_l <= 0.65:
        return "Medium"
    else:
        return "Dark"

def predict_skin_tone(image_folder, output_csv):
    """
    Predicts skin tones for all images in the specified folder and saves the results to a CSV file.
    Parameters:
        image_folder (str): Path to the folder containing images.
        output_csv (str): Path to the output CSV file.
    """
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    if not os.path.exists(image_folder):
        print(f"Error: The folder path '{image_folder}' does not exist.")
        return
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the specified folder.")
        return
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Skin Tone"])
        
        for image_name in image_files:
            image_path = os.path.join(image_folder, image_name)
            print(f"Processing {image_name}...")
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load image: {image_name}")
                    continue
                
                skin_tone = calculate_skin_tone(image, face_cascade)
                writer.writerow([image_name, skin_tone])
            
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

######################################################
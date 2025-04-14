# import cv2
# import mediapipe as mp
# import numpy as np
# def extract_mediapipe_attributes(image_path):
#     # Load and preprocess the image for Mediapipe
#     image = cv2.imread(image_path)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Initialize Mediapipe Face Mesh
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
#     results = face_mesh.process(rgb_image)
    
#     # Initialize dictionary for attributes
#     face_description = {}

#     def calculate_distance(point1, point2):
#         return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

#     # Extract attributes if landmarks are found
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract attributes as in the original Mediapipe code
#             left_eye_outer = face_landmarks.landmark[33]
#             left_eye_inner = face_landmarks.landmark[133]
#             left_eye_top = face_landmarks.landmark[159]
#             left_eye_bottom = face_landmarks.landmark[145]
#             left_eye_width = calculate_distance(left_eye_outer, left_eye_inner)
#             left_eye_height = calculate_distance(left_eye_top, left_eye_bottom)

#             right_eye_outer = face_landmarks.landmark[362]
#             right_eye_inner = face_landmarks.landmark[398]
#             right_eye_top = face_landmarks.landmark[386]
#             right_eye_bottom = face_landmarks.landmark[374]
#             right_eye_width = calculate_distance(right_eye_outer, right_eye_inner)
#             right_eye_height = calculate_distance(right_eye_top, right_eye_bottom)

#             left_eye_ratio = left_eye_width / left_eye_height
#             right_eye_ratio = right_eye_width / right_eye_height
#             face_description["eye_shape"] = "Almond-shaped" if left_eye_ratio > 1.8 and right_eye_ratio > 1.8 else "Round"

#             chin = face_landmarks.landmark[152]
#             left_cheekbone = face_landmarks.landmark[234]
#             right_cheekbone = face_landmarks.landmark[454]
#             forehead_center = face_landmarks.landmark[10]
#             jaw_width = calculate_distance(left_cheekbone, right_cheekbone)
#             face_height = calculate_distance(forehead_center, chin)

#             if jaw_width / face_height > 0.85:
#                 face_description["face_shape"] = "Square"
#             elif jaw_width / face_height > 0.75:
#                 face_description["face_shape"] = "Oval"
#             else:
#                 face_description["face_shape"] = "Heart"

#             left_mouth_corner = face_landmarks.landmark[61]
#             right_mouth_corner = face_landmarks.landmark[291]
#             upper_lip = face_landmarks.landmark[13]
#             lower_lip = face_landmarks.landmark[14]
#             mouth_width = calculate_distance(left_mouth_corner, right_mouth_corner)
#             mouth_height = calculate_distance(upper_lip, lower_lip)
#             face_description["mouth_shape"] = "Wide" if mouth_width / mouth_height > 2.0 else "Full"

#             cheek_x = int(left_cheekbone.x * image.shape[1])
#             cheek_y = int(left_cheekbone.y * image.shape[0])
#             cheek_region = image[cheek_y - 10:cheek_y + 10, cheek_x - 10:cheek_x + 10]
#             cheek_region_ycrcb = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2YCrCb)
#             avg_ycrcb = np.mean(cheek_region_ycrcb, axis=(0, 1))

#             if avg_ycrcb[1] > 145:
#                 face_description["skin_tone"] = "Fair"
#             elif avg_ycrcb[1] > 130:
#                 face_description["skin_tone"] = "Medium"
#             else:
#                 face_description["skin_tone"] = "Dark"
    
#     face_mesh.close()
#     return face_description

# def extract_mediapipe_attributes(image_path):
#     # Load and preprocess the image for Mediapipe
#     image = cv2.imread(image_path)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Initialize Mediapipe Face Mesh
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
#     results = face_mesh.process(rgb_image)
    
#     # Initialize dictionary for attributes
#     face_description = {}

#     def calculate_distance(point1, point2):
#         return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

#     # Extract attributes if landmarks are found
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract attributes as in the original Mediapipe code
#             left_eye_outer = face_landmarks.landmark[33]
#             left_eye_inner = face_landmarks.landmark[133]
#             left_eye_top = face_landmarks.landmark[159]
#             left_eye_bottom = face_landmarks.landmark[145]
#             left_eye_width = calculate_distance(left_eye_outer, left_eye_inner)
#             left_eye_height = calculate_distance(left_eye_top, left_eye_bottom)

#             right_eye_outer = face_landmarks.landmark[362]
#             right_eye_inner = face_landmarks.landmark[398]
#             right_eye_top = face_landmarks.landmark[386]
#             right_eye_bottom = face_landmarks.landmark[374]
#             right_eye_width = calculate_distance(right_eye_outer, right_eye_inner)
#             right_eye_height = calculate_distance(right_eye_top, right_eye_bottom)

#             left_eye_ratio = left_eye_width / left_eye_height
#             right_eye_ratio = right_eye_width / right_eye_height
#             face_description["eye_shape"] = "Almond-shaped" if left_eye_ratio > 1.8 and right_eye_ratio > 1.8 else "Round"

#             chin = face_landmarks.landmark[152]
#             left_cheekbone = face_landmarks.landmark[234]
#             right_cheekbone = face_landmarks.landmark[454]
#             forehead_center = face_landmarks.landmark[10]
#             jaw_width = calculate_distance(left_cheekbone, right_cheekbone)
#             face_height = calculate_distance(forehead_center, chin)

#             if jaw_width / face_height > 0.85:
#                 face_description["face_shape"] = "Square"
#             elif jaw_width / face_height > 0.75:
#                 face_description["face_shape"] = "Oval"
#             else:
#                 face_description["face_shape"] = "Heart"

#             left_mouth_corner = face_landmarks.landmark[61]
#             right_mouth_corner = face_landmarks.landmark[291]
#             upper_lip = face_landmarks.landmark[13]
#             lower_lip = face_landmarks.landmark[14]
#             mouth_width = calculate_distance(left_mouth_corner, right_mouth_corner)
#             mouth_height = calculate_distance(upper_lip, lower_lip)
#             face_description["mouth_shape"] = "Wide" if mouth_width / mouth_height > 2.0 else "Full"

#             # Improved skin tone analysis
#             cheek_x = int(left_cheekbone.x * image.shape[1])
#             cheek_y = int(left_cheekbone.y * image.shape[0])
#             cheek_region = image[max(cheek_y - 10, 0):min(cheek_y + 10, image.shape[0]), 
#                                  max(cheek_x - 10, 0):min(cheek_x + 10, image.shape[1])]
            
#             # Ensure the cheek region has valid data
#             if cheek_region.size > 0:
#                 # cheek_region_ycrcb = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2YCrCb)
#                 # avg_y, avg_cr, avg_cb = np.mean(cheek_region_ycrcb, axis=(0, 1))
                
#                 # # Normalize Cr and Cb values
#                 # normalized_cr = avg_cr / 255
#                 # normalized_cb = avg_cb / 255

#                 # # Skin tone determination based on Cr and Cb ranges
#                 # if normalized_cr > 0.6 and normalized_cb < 0.4:
#                 #     face_description["skin_tone"] = "Fair"
#                 # elif 0.45 < normalized_cr <= 0.6 and 0.3 < normalized_cb <= 0.4:
#                 #     face_description["skin_tone"] = "Medium"
#                 # else:
#                 #     face_description["skin_tone"] = "Dark"
#                  # Improved skin tone analysis using LAB color space
#                 cheek_region_lab = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2Lab)
#                 avg_l, avg_a, avg_b = np.mean(cheek_region_lab, axis=(0, 1))
                
#                 # Normalize LAB values
#                 normalized_l = avg_l / 255
#                 normalized_a = (avg_a - 128) / 128  # A and B range from 0-255, offset by 128
#                 normalized_b = (avg_b - 128) / 128

#                 # Skin tone classification
#                 if normalized_l > 0.7 and normalized_a < 0.15 and normalized_b < 0.15:
#                     face_description["skin_tone"] = "Fair"
#                 elif 0.5 <= normalized_l <= 0.7:
#                     face_description["skin_tone"] = "Medium"
#                 else:
#                     face_description["skin_tone"] = "Dark"

#     face_mesh.close()
#     return face_description

# ###################################################################

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from utils import skin
# from df import deepanalyse

def extract_mediapipe_attributes(image_path):
    # Load and preprocess the image for Mediapipe
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    results = face_mesh.process(rgb_image)

    # Initialize dictionary for attributes
    face_description = {}

    def calculate_distance(point1, point2):
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    # Extract attributes if landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eye attributes
            def extract_eye_features(landmark_indices):
                outer, inner, top, bottom = landmark_indices
                eye_outer = face_landmarks.landmark[outer]
                eye_inner = face_landmarks.landmark[inner]
                eye_top = face_landmarks.landmark[top]
                eye_bottom = face_landmarks.landmark[bottom]

                width = calculate_distance(eye_outer, eye_inner)
                height = calculate_distance(eye_top, eye_bottom)
                return width / height

            left_eye_ratio = extract_eye_features([33, 133, 159, 145])
            right_eye_ratio = extract_eye_features([362, 398, 386, 374])
            face_description["eye_shape"] = "Almond-shaped" if left_eye_ratio > 1.8 and right_eye_ratio > 1.8 else "Round"

            # Additional eye attributes
            face_description["eye_expression"] = "Wide-eyed" if left_eye_ratio > 2 else "Relaxed"

            # Skin tone and texture
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            skin_tone = skin.calculate_skin_tone(image, face_cascade)
            face_description["skin_tone"] = skin_tone

            # Lips and mouth
            mouth_width = calculate_distance(face_landmarks.landmark[61], face_landmarks.landmark[291])
            mouth_height = calculate_distance(face_landmarks.landmark[13], face_landmarks.landmark[14])
            face_description["lip_shape"] = "Heart-shaped" if mouth_width / mouth_height > 2.0 else "Full"

            # Nose attributes
            nose_bridge_height = calculate_distance(face_landmarks.landmark[6], face_landmarks.landmark[4])
            face_description["nose_shape"] = "Pointed" if nose_bridge_height < 0.1 else "Flat"

            # Facial symmetry
            symmetry_ratio = np.mean([
                calculate_distance(face_landmarks.landmark[234], face_landmarks.landmark[10]),
                calculate_distance(face_landmarks.landmark[454], face_landmarks.landmark[10])
            ])
            face_description["facial_symmetry"] = "Symmetric" if 0.95 <= symmetry_ratio <= 1.05 else "Asymmetric"

    face_mesh.close()

    print(f"mediapipe: {face_description}")
    objs = DeepFace.analyze(
        img_path=rf"{image_path}",
        actions=["age", "gender", "race"],
        enforce_detection=False,
    )
    objs[0].pop("gender")
    objs[0].pop("region")
    objs[0].pop("face_confidence")
    objs[0].pop("race")
    print(objs)
    # quit()
    face_description["deepface_attributes"] = objs[0]
    return face_description


################################################################

# import cv2
# import mediapipe as mp
# import numpy as np
# from deepface import DeepFace

# # DeepFace.build_model("Emotion")
# # DeepFace.build_model("Age")
# # DeepFace.build_model("Gender")
# # DeepFace.build_model("Race")

# def extract_mediapipe_attributes(image_path):
#     # Load and preprocess the image for Mediapipe
#     image = cv2.imread(image_path)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Initialize Mediapipe Face Mesh
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
#     results = face_mesh.process(rgb_image)

#     # Initialize dictionary for attributes
#     face_description = {}

#     def calculate_distance(point1, point2):
#         return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

#     # Extract attributes if landmarks are found
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Eye attributes
#             def extract_eye_features(landmark_indices):
#                 outer, inner, top, bottom = landmark_indices
#                 eye_outer = face_landmarks.landmark[outer]
#                 eye_inner = face_landmarks.landmark[inner]
#                 eye_top = face_landmarks.landmark[top]
#                 eye_bottom = face_landmarks.landmark[bottom]

#                 width = calculate_distance(eye_outer, eye_inner)
#                 height = calculate_distance(eye_top, eye_bottom)
#                 return width / height

#             left_eye_ratio = extract_eye_features([33, 133, 159, 145])
#             right_eye_ratio = extract_eye_features([362, 398, 386, 374])
#             face_description["eye_shape"] = "Almond-shaped" if left_eye_ratio > 1.8 and right_eye_ratio > 1.8 else "Round"
            
#             # Additional eye attributes
#             face_description["eye_expression"] = "Wide-eyed" if left_eye_ratio > 2 else "Relaxed"

#             # Skin tone and texture
#             cheek_x = int(face_landmarks.landmark[234].x * image.shape[1])
#             cheek_y = int(face_landmarks.landmark[234].y * image.shape[0])
#             cheek_region = image[max(cheek_y - 10, 0):min(cheek_y + 10, image.shape[0]), 
#                                  max(cheek_x - 10, 0):min(cheek_x + 10, image.shape[1])]

#             if cheek_region.size > 0:
#                 cheek_region_lab = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2Lab)
#                 avg_l, avg_a, avg_b = np.mean(cheek_region_lab, axis=(0, 1))

#                 normalized_l = avg_l / 255
#                 normalized_a = (avg_a - 128) / 128
#                 normalized_b = (avg_b - 128) / 128

#                 if normalized_l > 0.7 and normalized_a < 0.15 and normalized_b < 0.15:
#                     face_description["skin_tone"] = "Fair"
#                 elif 0.5 <= normalized_l <= 0.7:
#                     face_description["skin_tone"] = "Medium"
#                 else:
#                     face_description["skin_tone"] = "Dark"

#             # Lips and mouth
#             mouth_width = calculate_distance(face_landmarks.landmark[61], face_landmarks.landmark[291])
#             mouth_height = calculate_distance(face_landmarks.landmark[13], face_landmarks.landmark[14])
#             face_description["lip_shape"] = "Heart-shaped" if mouth_width / mouth_height > 2.0 else "Full"

#             # Nose attributes
#             nose_bridge_height = calculate_distance(face_landmarks.landmark[6], face_landmarks.landmark[4])
#             face_description["nose_shape"] = "Flat" if nose_bridge_height < 0.02 else "Pointed"

#             # Facial symmetry
#             symmetry_ratio = np.mean([
#                 calculate_distance(face_landmarks.landmark[234], face_landmarks.landmark[10]),
#                 calculate_distance(face_landmarks.landmark[454], face_landmarks.landmark[10])
#             ])
#             face_description["facial_symmetry"] = "Symmetric" if 0.95 <= symmetry_ratio <= 1.05 else "Asymmetric"

#     face_mesh.close()

#     # Use DeepFace to analyze additional attributes
#     try:
#         deepface_analysis = DeepFace.analyze(
#             img_path=image_path,
#             actions=["age", "gender", "race", "emotion"],
#             enforce_detection=True
#         )
#         face_description.update({
#             "age": deepface_analysis["age"],
#             "gender": deepface_analysis["gender"],
#             "race": deepface_analysis["dominant_race"],
#             "emotion": deepface_analysis["dominant_emotion"],
#         })
#     except Exception as e:
#         print(f"DeepFace analysis error: {e}")
#     print("mediapipe: {}".format(face_description))

#     return face_description

############################################################3

# import cv2
# import mediapipe as mp
# import numpy as np
# from deepface import DeepFace
# # from deepface.commons import functions
# from deepface import DeepFace
# # from deepface.basemodels import VGGFace
# import torch

# # # Ensure models are initialized properly
# # VGGFace.loadModel()
# # DeepFace.build_model("Emotion")
# # DeepFace.build_model("Age")
# # DeepFace.build_model("Gender")
# # DeepFace.build_model("Race")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device (GPU/CPU)
# print(f"Using device: {device}")

# def extract_mediapipe_attributes(image_path):
#     # Initialize DeepFace and clear cache
#     # functions.initialize_folder()

#     # Load image
#     image = cv2.imread(image_path)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Initialize Mediapipe Face Mesh
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
#     results = face_mesh.process(rgb_image)

#     # Initialize dictionary for attributes
#     face_description = {}

#     def calculate_distance(point1, point2):
#         return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

#     # Mediapipe facial attributes
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Example: Eye shape detection
#             def extract_eye_features(landmark_indices):
#                 outer, inner, top, bottom = landmark_indices
#                 eye_outer = face_landmarks.landmark[outer]
#                 eye_inner = face_landmarks.landmark[inner]
#                 eye_top = face_landmarks.landmark[top]
#                 eye_bottom = face_landmarks.landmark[bottom]
#                 width = calculate_distance(eye_outer, eye_inner)
#                 height = calculate_distance(eye_top, eye_bottom)
#                 return width / height

#             left_eye_ratio = extract_eye_features([33, 133, 159, 145])
#             right_eye_ratio = extract_eye_features([362, 398, 386, 374])
#             face_description["eye_shape"] = "Almond-shaped" if left_eye_ratio > 1.8 and right_eye_ratio > 1.8 else "Round"

#     # Close Mediapipe resources
#     face_mesh.close()

#     # DeepFace attributes
#     try:
#         deepface_analysis = DeepFace.analyze(
#             img_path=image_path,
#             actions=["age", "gender", "race", "emotion"],
#             enforce_detection=True
#         )
#         face_description.update({
#             "age": deepface_analysis["age"],
#             "gender": deepface_analysis["gender"],
#             "race": deepface_analysis["dominant_race"],
#             "emotion": deepface_analysis["dominant_emotion"],
#         })
#     except Exception as e:
#         print(f"DeepFace analysis error: {e}")
    
#     print("mediapipe: {}".format(face_description))
#     return face_description


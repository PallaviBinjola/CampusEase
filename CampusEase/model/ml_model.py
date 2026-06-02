import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import cv2

# ✅ Load pre-trained MobileNetV2 model (feature extractor)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# ✅ Extract feature vector from an image
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # resize to MobileNetV2 input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# ✅ Find the most similar image in a folder
def find_most_similar_image(uploaded_image_path, folder_path='static/img'):
    uploaded_features = extract_features(uploaded_image_path)
    if uploaded_features is None:
        return None

    best_match = None
    highest_similarity = -1

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            features = extract_features(image_path)
            if features is not None:
                # cosine similarity
                similarity = np.dot(uploaded_features, features) / (
                    np.linalg.norm(uploaded_features) * np.linalg.norm(features)
                )
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = file_name

    print(f"Best match: {best_match} (similarity: {highest_similarity})")
    return best_match

# ✅ Get top 3 most similar images (for recommendations)
def get_recommendations(img_path, folder_path='static/img', top_n=3):
    uploaded_features = extract_features(img_path)
    if uploaded_features is None:
        return []

    similarities = {}

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            features = extract_features(image_path)
            if features is not None:
                similarity = np.dot(uploaded_features, features) / (
                    np.linalg.norm(uploaded_features) * np.linalg.norm(features)
                )
                similarities[file_name] = similarity

    # Sort and return top N
    sorted_files = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_matches = [file for file, _ in sorted_files[:top_n]]

    print(f"Top {top_n} similar images: {top_matches}")
    return top_matches

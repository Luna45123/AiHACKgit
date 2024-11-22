import cv2
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import matplotlib.pyplot as plt
import json
import os

# Analyze moods for multiple faces in an image
def analyze_multiple_emotions(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Analyze moods using DeepFace
    analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # Ensure analysis is a list (it may be a dictionary if only one face is detected)
    if isinstance(analysis, dict):
        analysis = [analysis]

    # Process each face's analysis
    for idx, face_analysis in enumerate(analysis):
        mood = face_analysis['dominant_emotion']
        confidence = face_analysis['emotion'][mood]
        
        # Display the image with the face's mood
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Face {idx+1}: Mood: {mood} (Confidence: {confidence:.2f}%)")
        plt.axis('off')
        plt.show()

        # Log the mood to CSV and JSON
        log_mood(mood, confidence, face_analysis['emotion'], idx+1)

# Log mood details to CSV and JSON
def log_mood(mood, confidence, all_emotions, face_index):
    now = datetime.now()
    data = {
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "face_index": face_index,
        "mood": mood,
        "confidence": confidence,
        "all_emotions": all_emotions  # Detailed emotion scores
    }
    
    # Log to CSV
    df = pd.DataFrame([data])
    df.to_csv("multiple_mood_log.csv", mode='a', header=not os.path.exists("multiple_mood_log.csv"), index=False)

    # Log to JSON
    json_file = "multiple_mood_log.json"
    if os.path.exists(json_file):
        # Load existing data and append new entry
        with open(json_file, "r") as file:
            json_data = json.load(file)
        json_data.append(data)
    else:
        # Initialize with new data
        json_data = [data]
    
    # Save updated data to JSON
    with open(json_file, "w") as file:
        json.dump(json_data, file, indent=4)
    
    print(f"Mood for Face {face_index} logged successfully!")

# Run the multiple emotion detector
image_path = 'my2.jpg'  # Replace with your image path
analyze_multiple_emotions(image_path)


# import cv2
# import pandas as pd
# from deepface import DeepFace
# from datetime import datetime
# import matplotlib.pyplot as plt
# import json
# import os

# name = input('Enter your name: ')

# # Load an image and analyze mood
# def analyze_mood(image_path):
#     # Load the image
#     img = cv2.imread(image_path)

#     # Analyze mood using DeepFace
#     analysis = DeepFace.analyze(img, actions=['emotion','age'])
    
#     # Access the first result if analysis is a list
#     if isinstance(analysis, list):
#         analysis = analysis[0]

#     # Retrieve mood and confidence score
#     mood = analysis['dominant_emotion']
#     confidence = analysis['emotion'][mood]
#     age = analysis['age']

#     # Display the image and mood
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(f"{name} Mood: {mood} (Confidence: {confidence:.2f}%) {age} age: {age}")
#     plt.axis('off')
#     plt.show()

#     log_mood(mood, confidence, name,age)

# # Continue with the rest of the code...


# def log_mood(mood, confidence,name,age):
#     # Log date, mood, and confidence in a CSV file
#     now = datetime.now()
#     data = {
#         "date": [now.strftime("%Y-%m-%d %H:%M:%S")],
#         "mood": [mood],
#         "confidence": [confidence],
#         "name": [name],
#         "age": [age]
#     }
#     df = pd.DataFrame(data)
#     df.to_csv("mood_log.csv", mode='a', header=not pd.io.common.file_exists("mood_log.csv"), index=False)
#     print("Mood logged successfully!")

# # Log to JSON
#     json_file = "mood_log.json"
#     if os.path.exists(json_file):
#         # Load existing data and append new entry
#         with open(json_file, "r") as file:
#             json_data = json.load(file)
#         json_data.append(data)
#     else:
#         # Initialize with new data
#         json_data = [data]
    
#     # Save updated data to JSON
#     with open(json_file, "w") as file:
#         json.dump(json_data, file, indent=4)
    
#     print("Mood logged successfully!")

# # Run the mood tracker
# image_path = 'ag2.jpg'  # Replace with your image path
# analyze_mood(image_path)

# # from fer import FER
# # import cv2
# # import matplotlib.pyplot as plt

# # path = '2.jpg'

# # def analyze_mood(image_path):
# #     img = cv2.imread(image_path)
# #     detector = FER(mtcnn=True)
# #     emotion, score = detector.top_emotion(img)

# #     # Display image with mood
# #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #     plt.title(f"Mood: {emotion} (Confidence: {score*100:.2f}%)")
# #     plt.axis('off')
# #     plt.show()

# #     log_mood(emotion, score)

# # analyze_mood(path)
# # # Log mood as before...

import random
from flask import Flask, Response, jsonify, render_template, request
import cv2
from flask_cors import CORS, cross_origin
import openai
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import matplotlib.pyplot as plt
import json
import os
import logging
from textblob import TextBlob
from transformers import pipeline
from deep_translator import GoogleTranslator


from transformers import pipeline

logging.basicConfig(level=logging.DEBUG) 
data = {}

app = Flask(__name__)
app.run(port=5000)
CORS(app)





def analyze_multiple_emotions(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Analyze moods using DeepFace
    analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # Ensure analysis is a list (it may be a dictionary if only one face is detected)
    if isinstance(analysis, list):
        analysis = analysis[0]

    # Process each face's analysis
    mood = analysis['dominant_emotion']
    confidence = round(analysis['emotion'][mood], 2)
        
    # Log the mood to CSV and JSON
    mood_data = log_mood(mood, confidence, analysis['emotion'])
    return mood_data

# Log mood details to CSV and JSON
def log_mood(mood, confidence, all_emotions):
    all_emotions = {emotion: round(score, 2) for emotion, score in all_emotions.items()}
    now = datetime.now()
    global data 
    data = {
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "mood": mood,
        "confidence": confidence,
        "all_emotions": all_emotions  # Detailed emotion scores
    }
    #to do delete disgust and fear

    app.logger.info(data)

    # Log to CSV
    df = pd.DataFrame([data])
    df.to_csv("multiple_mood_log.csv", mode='a', header=not os.path.exists("multiple_mood_log.csv"), index=False)

    # Log to JSON
    json_file = "multiple_mood_log.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as file:
            json_data = json.load(file)
        json_data.append(data)
    else:
        json_data = [data]
    
    with open(json_file, "w") as file:
        json.dump(json_data, file, indent=4)
    
    return data



def translate_to_english(text):
    # Translate the text to English
    translation = GoogleTranslator(source='auto', target='en').translate(text)
    return translation

def analyze_mood(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


@app.route("/")
def home():
    return render_template("AI.html")

@app.route("/uploadMood", methods=["POST"])
def upload_mood():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)  # Ensure the 'temp' directory exists
    file.save(file_path)

    try:
        mood_data = analyze_multiple_emotions(file_path)
    finally:
        print('save')
        # os.remove(file_path)

    return jsonify(mood_data), 200

@app.route("/getMood")
@cross_origin()
def getMood():
    # Run the multiple emotion detector
    img = ['sad.jpg','my2.jpg','ag.jpg','test.jpg']
    image_path = random.choice(img)
  # Replace with your image path
    analyze_multiple_emotions(image_path)
    print(data, image_path)
    json_data2 = json.dumps(data)

    # return Response(f"<h1>Hello World<h1> <p>test<P> <p>{data}  {image_path}<p></br> ",status=200)
    return Response(json_data2, status=200, mimetype='application/json')


    

@app.route('/getMoodText', methods=['GET'])
def moodText():
    text = request.args.get('text')
    if text:
        mood = analyze_mood(text)
        return jsonify({"mood": mood})
    else:
        return jsonify({"error": "No text provided"}), 400

@app.route('/getTextFrom')
def getText():
    return render_template("AI.html")

@app.route('/getMoodText2', methods=['POST'])
def moodText2():
    data = request.get_json()  # Parse JSON payload
    text = data.get('text', '')
      # Get 'text' from the JSON body
    if len(text) > 1800:
        text = text[:1800]
    translated_text = translate_to_english(text)
    text = translated_text
    
    if text:
        mood = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        result = mood(text)
        response = {
            # "translated_text": translated_text,
            "mood_analysis": result
        }
        return jsonify(response)
    else:
        return jsonify({"error": "No text provided"}), 400
    

@app.route('/c')
def c():
    return render_template("testgraph.html")

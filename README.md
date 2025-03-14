# PlantCare.AI
Plant Disease Prediction and IoT-Based Soil Monitoring System 🌱🌍
📌 Overview

Plant Care AI is an advanced IoT-based system that predicts plant diseases using deep learning models and monitors essential environmental parameters such as temperature, humidity, and soil pH. This system integrates IoT sensors with Firebase for real-time data collection and visualization. Users can upload plant leaf images via a Streamlit-based web application to detect diseases and analyze environmental conditions.

![image](https://github.com/user-attachments/assets/85d2e811-1b77-41ed-bdae-94b6a3ce2918)

🛠️ Features
✅ Plant Disease Prediction – Upload an image of a plant leaf to detect diseases using a trained TensorFlow/Keras model.

✅ IoT-Based Monitoring – Collect real-time data from sensors measuring temperature, humidity, soil pH, etc.

✅ Firebase Integration – Sensor data is stored and retrieved from Firebase for seamless analysis.

✅ Streamlit Web Application – A user-friendly interface to upload images, visualize sensor data, and track plant health trends.

✅ Statistical Analysis – View graphs and insights on environmental conditions affecting plant health.

Block Diagram :
![image](https://github.com/user-attachments/assets/f6452266-96f3-4009-91e4-cea310bb85e4)

Kaggle Dataset URL : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

About Dataset : -This dataset consists of more than 87,000 images each of healthy and diseased plant leaves and was used for the development of the CNN model. The dataset is split into training and testing sets in an 80:20 ratio.

📡 IoT Integration
This project uses IoT sensors to collect data and send it to Firebase. The following sensors are used:

🔹 DHT11 – Measures temperature & humidity

🔹 Soil Moisture Sensor – Detects soil moisture levels

🔹 pH Sensor – Measures soil acidity

🔹 Light Intensity Sensor (BH1750) – Monitors sunlight exposure

Data Flow
1️⃣ ESP8266/NodeMCU reads sensor data.

2️⃣ Data is sent to Firebase Realtime Database.

3️⃣ Streamlit app retrieves and visualizes the data.

🛠️ Technologies Used
🔹 Python (TensorFlow, Streamlit, Pandas, OpenCV, NumPy)

🔹 Firebase (Realtime Database)

🔹 IoT Sensors (DHT11, Soil Moisture, pH, BH1750)

🔹 Machine Learning (Deep Learning Model for Disease Prediction)

🔹 ESP8266/NodeMCU (For Sensor Data Collection)

🔗 Firebase Setup
Create a Firebase Project on Firebase Console.
Download the JSON credentials file and place it in the project directory (firebase.json).
Ensure Firebase Admin SDK is properly configured in app.py.

 >>Steps to Run the Project :
1] Clone the repository :
git clone https://github.com/vishwajeet-barade/Plant-Care-AI.git

cd Plant-Care-AI

2]pip install -r requirements.txt
3]streamlit run app.py



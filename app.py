import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
import numpy as np
import tensorflow as tf

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

# Initialize Firebase connection
def init_firebase():
    cred_path = "dht11-9954d-firebase-adminsdk-59f8a-a7e7b9d4e6.json"  # Replace with your Firebase credentials
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://dht11-9954d-default-rtdb.firebaseio.com/'  # Replace with your Firebase database URL
        })

# Disease Solutions Dictionary
disease_solutions = {
    'Apple___Apple_scab': "Apply fungicides like captan or myclobutanil. Ensure proper pruning and removal of infected leaves.",
    'Apple___Black_rot': "Prune and destroy infected plant parts. Apply copper-based fungicides during dormant periods.",
    'Apple___Cedar_apple_rust': "Remove nearby cedar trees if possible. Use resistant apple varieties and apply fungicides.",
    'Apple___healthy': "No action needed; the plant is healthy.",
    'Blueberry___healthy': "No action needed; the plant is healthy.",
    'Cherry_(including_sour)___Powdery_mildew': "Apply sulfur or potassium bicarbonate fungicides. Ensure proper air circulation.",
    'Cherry_(including_sour)___healthy': "No action needed; the plant is healthy.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Apply fungicides like azoxystrobin. Rotate crops to prevent pathogen buildup.",
    'Corn_(maize)___Common_rust_': "Use resistant varieties. Apply fungicides like mancozeb if severe.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant hybrids and apply fungicides like propiconazole when necessary.",
    'Corn_(maize)___healthy': "No action needed; the plant is healthy.",
    'Grape___Black_rot': "Prune infected areas and apply fungicides like mancozeb during the growing season.",
    'Grape___Esca_(Black_Measles)': "Avoid pruning in wet conditions. Remove severely infected vines and maintain proper nutrition.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use copper-based fungicides and improve air circulation around plants.",
    'Grape___healthy': "No action needed; the plant is healthy.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees and control psyllid vectors with insecticides.",
    'Peach___Bacterial_spot': "Apply copper-based sprays. Avoid overhead watering and use resistant varieties.",
    'Peach___healthy': "No action needed; the plant is healthy.",
    'Pepper,_bell___Bacterial_spot': "Apply copper-based bactericides. Avoid working with wet plants to reduce spread.",
    'Pepper,_bell___healthy': "No action needed; the plant is healthy.",
    'Potato___Early_blight': "Apply fungicides like chlorothalonil. Rotate crops and remove debris after harvest.",
    'Potato___Late_blight': "Use resistant varieties and apply fungicides like fluopicolide. Avoid overhead irrigation.",
    'Potato___healthy': "No action needed; the plant is healthy.",
    'Raspberry___healthy': "No action needed; the plant is healthy.",
    'Soybean___healthy': "No action needed; the plant is healthy.",
    'Squash___Powdery_mildew': "Apply sulfur-based fungicides and ensure adequate spacing between plants.",
    'Strawberry___Leaf_scorch': "Remove infected leaves and apply fungicides like captan. Maintain proper irrigation.",
    'Strawberry___healthy': "No action needed; the plant is healthy.",
    'Tomato___Bacterial_spot': "Use resistant varieties and apply copper-based sprays. Avoid overhead watering.",
    'Tomato___Early_blight': "Apply fungicides like chlorothalonil. Rotate crops and use resistant varieties.",
    'Tomato___Late_blight': "Apply fungicides like mancozeb and avoid overhead irrigation.",
    'Tomato___Leaf_Mold': "Use resistant varieties and ensure proper ventilation in greenhouses.",
    'Tomato___Septoria_leaf_spot': "Prune affected leaves and apply fungicides like chlorothalonil.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides or natural predators like ladybugs.",
    'Tomato___Target_Spot': "Apply fungicides like chlorothalonil and maintain good field hygiene.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whiteflies with insecticides and use resistant varieties.",
    'Tomato___Tomato_mosaic_virus': "Use virus-free seeds and practice good sanitation.",
    'Tomato___healthy': "No action needed; the plant is healthy."
}

# Function to upload and display an image
def upload_image():
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload a plant leaf image", type=["png", "jpg", "jpeg"])
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
                      
        predicted_class = class_name[result_index]
        solution = disease_solutions[predicted_class]
        st.success("Model is Predicting it's a {}".format(predicted_class))
        st.info("Recommended Solution: {}".format(solution))

# Function to display data from Firebase
def show_data():
    st.write("Fetching data from Firebase...")

    try:
        # Reference to the database
        humidity_ref = db.reference('DHT/humidity')
        temperature_ref = db.reference('DHT/temperature')
        soil_moisture_ref = db.reference('SoilMoisture/percentage')  # Reference to the soil moisture data
        
        # Fetching data from Firebase
        latest_humidity = humidity_ref.get()  # Single float value
        latest_temperature = temperature_ref.get()  # Single float value
        latest_soil_moisture = soil_moisture_ref.get()  # Single float value (percentage)

        if latest_humidity is None or latest_temperature is None or latest_soil_moisture is None:
            st.error("No data found in Firebase.")
            return

        # Display latest data
        st.write("Latest Humidity, Temperature, and Soil Moisture Values:")
        st.write(f"Humidity: {latest_humidity}%")
        st.write(f"Temperature: {latest_temperature}°C")
        st.write(f"Soil Moisture: {latest_soil_moisture}%")

        # Simulate historical data for charts
        # Replace this with real historical data if available
        data = {
            "Timestamp": pd.date_range(start="2024-01-01", periods=10, freq="H"),  # Example timestamps
            "Humidity": [latest_humidity] * 10,  # Simulated 10 readings
            "Temperature": [latest_temperature] * 10,  # Simulated 10 readings
            "Soil Moisture": [latest_soil_moisture] * 10  # Simulated 10 readings for soil moisture
        }
        data_df = pd.DataFrame(data)

        # Line Chart for Temperature
        st.subheader("Temperature Over Time")
        st.line_chart(data_df.set_index("Timestamp")["Temperature"])

        # Line Chart for Humidity
        st.subheader("Humidity Over Time")
        st.line_chart(data_df.set_index("Timestamp")["Humidity"])

        # Line Chart for Soil Moisture
        st.subheader("Soil Moisture Over Time")
        st.line_chart(data_df.set_index("Timestamp")["Soil Moisture"])
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")


# Main function for the Streamlit app
def main():
    st.set_page_config(layout="wide")

    # Navbar using streamlit-option-menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload", "Data", "About"],
        icons=["house", "cloud-upload", "table", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    # Home page
    if selected == "Home":
        st.markdown("""
            <style>
                .separator {
                margin-top: 40px;
                margin-bottom: 40px;
                border-top: 2px solid #cccccc;
            </style>                
           <div style="text-align: center;">
                <h1 style="color: red;">Welcome to PlantCare.AI</h1>
           </div>
           <div style="text-align: center;">
                <h4 >Use the Navigator bar to upload image and Anlysis Data . </h4>
           </div>
           <div class='separator'></div>                    
                        
            """, unsafe_allow_html=True)
        st.markdown("""
             <style>
            .head-start{
                display:flex;
                gap:20px;
                margin-bottom:20px;
                padding-left:15px;
                background-color:#e8e7e3;
            }
            .message{
                width:45%;
            }
            .message h3{
                font-size:25px;
            }
            .message p{
                font-size:16px;
            }
            .head-image{
                display:flex;
                gap:15px;
            }
            .head-image img{
                height:280px;
                width:350px;
            }
            .image-text{
                width:35%;
                padding:20px;
                margin:30px;
                font-size:20px;
            }
            .image-text span{
                font-weight:bold;
                font-size:25px;
                color: #eb5e34;
            }
            .demo-text{
                width:100%;
                display:flex;
                justify-content:center;
                font-size:23px;
                color:#eb5e34;
                margin-bottom:20px;
            }
            .image-row {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .image-row img {
                width: 30%;
                height:250px;
                border-radius: 8px;
            }
            .head2-msg{
                display:flex;
                flex-direction:column;
                gap:20px;
                justify-content:center;
                align-item:center;
                margin-top:15px;
                margin-bottom:25px;
            }
            .head2-text{
                display:flex;
                flex-direction:column;
                gap:20px;
            }
            .head2-image{
                display:flex;
                gap:20px;
            }
            .head2-image1{
                display:flex;
                flex-direction:column;
                border-radius:5px;
                border:2px solid #e8e7e3;     
            }
            .head2-image2{
               display:flex;
               flex-direction:column;
               border-radius:5px;
               border:2px solid #e8e7e3; 
            }
            .head2-image1-img{
                display:flex;
                width:700px;
            }
            .head2-image2-img{
                display:flex;
                width:700px;
            }
            .head2-image1-img img{
                height:280px;
                width:350px;
            }
            .head2-image2-img img{
                height:280px;
                width:350px;
            }
            .head2-image1-text{
                padding:15px;
            }
            .head2-image2-text{
                padding:15px;
            }
            .head3-msg-img{
                display:flex;
                gap:40px;
                margin-top:25px;
                background-color:#e8e7e3;
                margin-bottom:20px;
            }
            .head3-msg{
                width:30%;
                height:300px;
                display:flex;
                flex-direction:column;
                gap:20px;
                margin-top:25px;
                justify-content:ceter;
                align-item:center;
            }
             </style>
       <div class='head-start'>
          <div class='message'>
             <h3>How to Colorize a Black and White Photo</h3>
          <div class='msg-info'>
            <p>1.Upload image: Upload image to Leaf plant Detection.</p>
            <p>2.Automated Colorization: The AI will automatically analyze and detect plant disease.</p>
            <p>3.Analysis Data :Analyse the Data of your Soil using our model.</p>
            <p>4.Get Solution : Get reccommendation to enhance crop.</p>
          </div>
          </div>
          <div class='head-image'>
          <img src="https://cdn.pixabay.com/photo/2012/10/06/02/22/gymnosporangium-fuscum-59906_960_720.jpg" alt="Image 1">
          <img src="https://cdn.pixabay.com/photo/2012/09/04/20/44/wine-leaf-56053_640.jpg" alt="Image 2">
          </div>
       </div>
       <div class='head2-msg'>
          <div class='head2-text'>
            <h2>Let's see the results for plant Diseased and Healthy image</h2>
            <p>Fotor allows your colorize images of various types, from the family old photos to the black and white celebrity photos. 
            With less than three seconds, those precious memories will be vibrant and alive in our online photo colorizer!</p>
          </div>
          <div class='head2-image'>
             <div class='head2-image1'>
                <div class='head2-image1-img'>
                    <img src="https://cdn.pixabay.com/photo/2016/09/02/07/17/autumn-1638473_640.jpg" alt="Image 1">
                    <img src="https://cdn.pixabay.com/photo/2015/08/14/19/50/maple-888807_640.jpg" alt="Image 2">
                </div>
                <div class='head2-image1-text'>
                <h4>Colorize:Breathe Color into Old Photos Effortlessly</h4>
                <p>We do more than just add hues ; we rejuvenate old photos,repair damage,and restore faded details,enhancing their quality to make 
                them appear as if captured just yesterday. </p>
                </div>
             </div>
             <div class='head2-image2'>
                 <div class='head2-image2-img'>
                    <img src="https://cdn.pixabay.com/photo/2018/10/18/00/34/leaf-3755340_640.jpg" alt="Image 1">
                    <img src="https://cdn.pixabay.com/photo/2016/08/30/16/05/leaf-1631181_640.jpg" alt="Image 2">
                </div>
                <div class='head2-image2-text'>
                 <h4>Colorize Photos to Make History Come Alive </h4>
                 <p>If you are curious about how amazing and marvelous the history is , try photo colorizer to magically add color to black and 
                 white photos that witness you want to colorize clasic movie screenshots, historical figures or black and white photos of landscape
                 and building , we have you covered.</p>
                </div>
             </div>
          </div>
       </div>
     """,unsafe_allow_html=True)
        

    # Upload page
    elif selected == "Upload":
        st.subheader("Upload Leaf Image")
        upload_image()

    # Data page
    elif selected == "Data":
        st.subheader("Data from Firebase")
        init_firebase()  # Initialize Firebase connection
        show_data()  # Fetch and display data from Firebase

    # About page
    elif selected == "About":
        st.subheader("About This Project")
        st.write("This project is built using Streamlit to interact with a machine learning model for plant disease prediction.")
        st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)


if __name__ == "__main__":
    main()


 # Add a footer
st.markdown("""
<style>
   .footer{
     background-color:#383837;
     color:#ffffff;
     padding:30px;
   }
   .footer1{
      display:flex;
      justify-content: space-evenly;
      margin-top:50px
      margin-bottom:40px
   }
   .footer2{
    display:flex;
    justify-content:center;
    margin-top:20px;
   }
</style>

<div class='footer'>
   <div class='footer1'>
      <div class='common'>
       <p class='common-head'>Company</p>
       <p>About Us</p>
       <p>Privacy Policy</p>
       <p>Terms of Service</p>
       <p>Contact Us</p>
      </div>
     <div class='common'>
       <p class='common-head'>Support</p>
       <p>Help Center</p>
       <p>Blog & Tutorial</p>
       <p>Pricing</p>
       <p>Product Updates</p>
     </div>
     <div class='common'>
       <p class='common-head'>Platforms</p>
       <p> for Windows</p>
     </div>
     <div class='common'>
       <p class='common-head'>Resource</p>
       <p>White Background</p>
       <p>Black Background</p>
       <p>Aesthetic Background</p>
       <p>Flower Background</p>
     </div>

   </div>
   <div class='footer2'><p>© 2024 Everimaging, All Rights Reserved.</p></div>
</div>
""", unsafe_allow_html=True)
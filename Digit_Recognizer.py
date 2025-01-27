import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageDraw
import requests
import io

# Dynamically load the model and scaler
model_url = "https://drive.google.com/uc?export=download&id=1dCvnlhpxrUlMX-L8_u1JvDXWbMeYrrGs"
scaler_url = "https://drive.google.com/uc?export=download&id=1J2LlO3LLAG2Cjle_EV4bB39IYhKanJGm"

# Load model
model_response = requests.get(model_url)
model = joblib.load(io.BytesIO(model_response.content))

# Load scaler
scaler_response = requests.get(scaler_url)
scaler = joblib.load(io.BytesIO(scaler_response.content))

st.title("MNIST Digit Classifier")

# Set up the canvas for drawing digits
canvas = st_canvas(
    fill_color="white",
    stroke_color="black",
    stroke_width=20,
    width=280,
    height=280,
    drawing_mode="freedraw",
    update_streamlit=True,
    key="canvas"
)

def preprocess_and_predict(canvas_data):
    if canvas_data and canvas_data["objects"]:
        # Create a blank white image
        img = Image.new("L", (280, 280), "white")
        draw = ImageDraw.Draw(img)

        # Draw paths from canvas JSON data
        for obj in canvas_data["objects"]:
            if obj["type"] == "path":
                path = obj["path"]
                for p in range(len(path) - 1):
                    start = (path[p][1], path[p][2])
                    end = (path[p + 1][1], path[p + 1][2])
                    draw.line([start, end], fill="black", width=10)

        # Crop the image to the bounding box of the digit
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Resize to 28x28, maintaining aspect ratio and centering
        img = ImageOps.pad(img, (28, 28), color="black")

        # Invert colors (black on white to white on black)
        img = ImageOps.invert(img)

        # Display the image the model will use for prediction (28x28)
        st.image(img, caption="Image Model Thinks About (28x28)", use_container_width=False)

        # Convert image to NumPy array and flatten it
        img_array = np.array(img).reshape(1, -1)

        # Scale the image data
        img_scaled = scaler.transform(img_array)

        # Predict the digit using the trained model
        prediction = model.predict(img_scaled)
        return prediction[0]

    st.write("No valid image data detected. Please draw a digit.")
    return None

# Button for triggering the prediction
if st.button("Predict"):
    prediction = preprocess_and_predict(canvas.json_data)
    if prediction is not None:
        st.write(f"Predicted digit: {prediction}")
    else:
        st.write("Please draw a digit before predicting.")

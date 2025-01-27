import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageDraw
import requests
import io

def load_model():
    try:
        model_url = "https://github.com/HassanDaher93/mnist-digit-classifier/releases/download/stacking_model/stacking_classifier_model.pkl"
        response = requests.get(model_url, timeout=10)  # Add timeout to prevent hanging

        if response.status_code == 200:
            # If download succeeds, load the model
            model = joblib.load(io.BytesIO(response.content))
            st.write("Model loaded successfully!")
            return model
        else:
            st.error(f"Error downloading model. HTTP Status: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch network-related errors
        st.error(f"Network error: {e}")
        return None
    except Exception as e:
        # Catch all other exceptions
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()



# Load model and handle potential errors
model = load_model()
if model is None:
    st.error("Model could not be loaded. Please check the URL and try again.")

# Load the scaler (assuming it's a local file)
scaler = joblib.load("scaler.pkl")

# Title of the app
st.title("MNIST Digit Classifier")

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. **Draw a digit (0-9)** on the canvas.
2. **Click "Predict"** to see the model's prediction.
3. Make sure your digit is clear and fills the canvas for best results.
""")

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

# Function to preprocess the image and predict the digit
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

# Buttons for interacting with the app
if st.button("Predict"):
    prediction = preprocess_and_predict(canvas.json_data)
    if prediction is not None:
        st.write(f"Predicted digit: {prediction}")
    else:
        st.write("Please draw a digit before predicting.")

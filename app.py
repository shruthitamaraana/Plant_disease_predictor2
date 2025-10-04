import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Replicating the Old UI using HTML & CSS ---
# We will inject a large block of CSS and the HTML for the background icons.

def load_css_and_html():
    """Injects custom CSS and the HTML for the background icons."""
    css = """
    <style>
        /* --- Hide Streamlit's default elements --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .st-emotion-cache-18ni7ap { /* Main container padding */
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp {
             background-color: transparent;
        }

        /* --- Global Styles from Flask App --- */
        body {
            font-family: 'Poppins', sans-serif !important;
            background-color: #f8f9fa; /* Apply background color to the body */
        }

        h1, h2, h3, h4, p, div, span, label {
            font-family: 'Poppins', sans-serif !important;
        }

        /* --- Custom Containers to Mimic Flask Layout --- */
        .container {
            width: 100%;
            max-width: 900px;
            margin: auto;
        }

        .custom-header {
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #dee2e6;
        }

        .custom-header h1 {
            font-size: 3.2rem;
            font-weight: 700;
            color: #2d6a4f;
        }

        .custom-header p {
            font-size: 1.15rem;
            color: #6c757d;
            max-width: 600px;
            margin: 0.5rem auto 0;
        }
        
        /* --- Card Styling for Uploader and Reset Button --- */
        .card {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(149, 157, 165, 0.15);
            margin-bottom: 2.5rem;
            border: 1px solid #e9ecef;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed #ced4da;
            padding: 1rem;
        }

        .reset-button-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .stButton>button {
            border-radius: 50px !important;
            font-weight: 600 !important;
            padding: 0.8rem 2rem !important;
            width: auto;
        }

        /* --- "How It Works" Section --- */
        .how-it-works {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2.5rem;
        }
        .how-it-works h2 {
            font-size: 2rem;
            font-weight: 600;
            color: #2d6a4f;
            margin-bottom: 2rem;
        }
        
        .step { padding: 1.5rem; text-align: center; }
        .step-icon {
            width: 70px; height: 70px;
            background-color: #e7f5ec;
            border-radius: 50%;
            display: flex; justify-content: center; align-items: center;
            margin: 0 auto 1.5rem;
            color: #40916c;
        }
        .step h3 { font-weight: 600; margin-bottom: 0.5rem; font-size: 1.25rem;}
        .step p { color: #6c757d; max-width: 250px; margin: 0 auto; }

        /* --- Results Section --- */
        .results-card {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(149, 157, 165, 0.15);
            margin-bottom: 2.5rem;
            border: 1px solid #e9ecef;
        }
        .results-card h3 {
             margin-top: 0; font-weight: 600; color: #212529;
             padding-bottom: 1rem; margin-bottom: 1.5rem;
             border-bottom: 1px solid #e9ecef;
        }
        /* Simplified rule, as resizing is now handled in Python */
        .results-card div[data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid #e9ecef;
        }
        .main-prediction {
            background-color: #f8f9fa; padding: 1.5rem; border-radius: 12px;
            border-left: 5px solid #40916c; font-size: 1.1rem; margin-bottom: 1.5rem;
        }
        .main-prediction strong { font-size: 1.4rem; color: #2d6a4f; display: block; }
        
        .healthy-info {
             border: 1px solid #b7e4c7; background-color: #e7f5ec; border-radius: 12px;
             padding: 1.5rem; margin-bottom: 1.5rem;
        }
        .healthy-info h4 { color: #2d6a4f; margin-top: 0; }
        
        .disease-info {
             border: 1px solid #ffe8cc; background-color: #fff4e6; border-radius: 12px;
             padding: 1.5rem; margin-bottom: 1.5rem;
        }
        .disease-info h4 { color: #d9534f; margin-top: 0; }
        
        .top-predictions ul { list-style-type: none; padding: 0; }
        .top-predictions li { padding: 0.75rem 0; border-bottom: 1px solid #e9ecef; display: flex; justify-content: space-between; }
        
        /* --- Floating Background Icons --- */
        .background-icons {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            overflow: hidden; z-index: -1;
        }
        .background-icons .icon {
            position: absolute; display: block;
            background-color: rgba(64, 145, 108, 0.15);
            border-radius: 50%;
            animation: float 25s linear infinite;
            bottom: -150px;
        }
        .background-icons .icon:nth-child(1) { left: 25%; width: 80px; height: 80px; animation-delay: 0s; }
        .background-icons .icon:nth-child(2) { left: 10%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
        .background-icons .icon:nth-child(3) { left: 70%; width: 20px; height: 20px; animation-delay: 4s; }
        .background-icons .icon:nth-child(4) { left: 40%; width: 60px; height: 60px; animation-delay: 0s; animation-duration: 18s; }
        .background-icons .icon:nth-child(5) { left: 65%; width: 20px; height: 20px; animation-delay: 0s; }
        .background-icons .icon:nth-child(6) { left: 75%; width: 110px; height: 110px; animation-delay: 3s; }
        .background-icons .icon:nth-child(7) { left: 35%; width: 150px; height: 150px; animation-delay: 7s; }
        .background-icons .icon:nth-child(8) { left: 50%; width: 25px; height: 25px; animation-delay: 15s; animation-duration: 45s; }
        .background-icons .icon:nth-child(9) { left: 20%; width: 15px; height: 15px; animation-delay: 2s; animation-duration: 35s; }
        .background-icons .icon:nth-child(10) { left: 85%; width: 150px; height: 150px; animation-delay: 0s; animation-duration: 11s; }
        @keyframes float {
            0% { transform: translateY(0) rotate(0deg); opacity: 1; }
            100% { transform: translateY(-1000px) rotate(720deg); opacity: 0; }
        }
    </style>
    """
    
    html_background = """
    <div class="background-icons">
        <div class="icon"></div> <div class="icon"></div> <div class="icon"></div>
        <div class="icon"></div> <div class="icon"></div> <div class="icon"></div>
        <div class="icon"></div> <div class="icon"></div> <div class="icon"></div>
        <div class="icon"></div>
    </div>
    """
    
    # Inject Fonts and styles
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(html_background, unsafe_allow_html=True)

# Call the function to apply styles
load_css_and_html()

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = 'model.keras'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    try:
        # --- MODIFIED: Re-create the model architecture using the Functional API ---
        
        # 1. Define the input layer
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        # 2. Add the missing Normalization layer as indicated by the error
        x = tf.keras.layers.Normalization(name='normalization')(inputs)

        # 3. Define the base model (EfficientNetB2)
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights=None,
            input_tensor=x # Connect base model to the normalization layer
        )
        base_model.trainable = False

        # 4. Define the custom layers and connect them
        y = base_model.output
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Dropout(0.4)(y)
        y = tf.keras.layers.Dense(512, activation='relu')(y)
        y = tf.keras.layers.Dropout(0.3)(y)
        outputs = tf.keras.layers.Dense(38, activation='softmax')(y)

        # 5. Create the final model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # 6. Load the saved weights into the newly defined model structure
        model.load_weights(model_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("This might be due to an architecture mismatch. Ensure the defined architecture here matches the one in your training script.")
        st.stop()

model = load_model()

# --- Class Names and Disease Info ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
    'Tomato___healthy'
]
DISEASE_INFO = {
    'Apple - Apple scab': { 'description': "Apple scab is a fungal disease that results in brown or black spots on leaves, fruit, and twigs.", 'treatment': "Apply fungicides, prune infected areas, and ensure good air circulation." },
    'Apple - Black rot': { 'description': "Black rot is a fungal disease that causes dark, rotten spots on apples, often starting at the blossom end.", 'treatment': "Prune out cankers, remove mummified fruit, and apply appropriate fungicides." },
    'Apple - Cedar apple rust': { 'description': "Cedar apple rust is a fungal disease requiring both apple and cedar trees to complete its life cycle. It causes yellow-orange spots on leaves.", 'treatment': "Remove nearby cedar trees if possible, or apply fungicides during the infection period." },
    'Cherry (including sour) - Powdery mildew': { 'description': "Powdery mildew is a fungal disease that appears as white powdery spots on leaves and shoots.", 'treatment': "Apply fungicides, improve air circulation, and remove infected plant parts." },
    'Corn (maize) - Cercospora leaf spot Gray leaf spot': { 'description': "Gray leaf spot is a fungal disease of corn that causes long, rectangular, grayish lesions on the leaves.", 'treatment': "Use resistant hybrids, practice crop rotation, and apply fungicides." },
    'Corn (maize) - Common rust': { 'description': "Common rust is a fungal disease characterized by small, reddish-brown pustules on both sides of the leaves.", 'treatment': "Resistant hybrids are the primary management tool. Fungicides can be effective if applied early." },
    'Corn (maize) - Northern Leaf Blight': { 'description': "Northern Leaf Blight is a fungal disease that produces long, elliptical, grayish-green or tan lesions on corn leaves.", 'treatment': "Plant resistant hybrids, use tillage to bury crop residue, and apply fungicides when necessary." },
    'Grape - Black rot': { 'description': "Black rot is a serious fungal disease of grapes, causing dark spots on leaves, shoots, and fruit.", 'treatment': "Apply fungicides on a regular schedule, practice good sanitation, and improve air circulation." },
    'Grape - Esca (Black Measles)': { 'description': "Esca is a complex fungal disease that causes tiger-stripe patterns on leaves and can lead to vine death.", 'treatment': "Management is difficult. Prune out dead wood and consider surgical trunk renewal." },
    'Grape - Leaf blight (Isariopsis Leaf Spot)': { 'description': "This disease causes dark brown, angular spots on grape leaves, which can merge and cause defoliation.", 'treatment': "Fungicide applications are effective. Ensure good vineyard sanitation." },
    'Orange - Haunglongbing (Citrus greening)': { 'description': "Citrus greening is a devastating bacterial disease that results in mottled leaves, misshapen fruit, and eventual tree death.", 'treatment': "There is no cure. Management focuses on removing infected trees and controlling the insect vector (Asian citrus psid)." },
    'Peach - Bacterial spot': { 'description': "Bacterial spot causes dark, angular lesions on leaves and sunken spots on fruit.", 'treatment': "Use resistant varieties, apply bactericides, and manage tree nutrition." },
    'Pepper, bell - Bacterial spot': { 'description': "This bacterial disease causes water-soaked spots on leaves and raised, scab-like spots on fruit.", 'treatment': "Use disease-free seed, practice crop rotation, and apply copper-based bactericides." },
    'Potato - Early blight': { 'description': "Early blight is a fungal disease that causes dark, 'target-like' spots on lower, older leaves.", 'treatment': "Apply fungicides, practice crop rotation, and ensure proper plant nutrition." },
    'Potato - Late blight': { 'description': "Late blight is a destructive fungal disease that causes large, dark lesions on leaves and can rapidly destroy a crop.", 'treatment': "Requires aggressive fungicide applications. Use certified seed potatoes and remove volunteer plants." },
    'Squash - Powdery mildew': { 'description': "Powdery mildew appears as white, powdery spots on the leaves and stems of squash plants.", 'treatment': "Apply fungicides (including organic options like sulfur or neem oil) and plant resistant varieties." },
    'Strawberry - Leaf scorch': { 'description': "Leaf scorch is a fungal disease that causes dark purple, irregular-shaped blotches on strawberry leaves.", 'treatment': "Use resistant varieties, remove old leaves after harvest, and apply fungicides." },
    'Tomato - Bacterial spot': { 'description': "Bacterial spot on tomatoes causes small, water-soaked spots on leaves and fruit.", 'treatment': "Start with clean seed, practice crop rotation, and use copper sprays." },
    'Tomato - Early blight': { 'description': "Early blight is a common fungal disease causing 'target' spots on lower leaves.", 'treatment': "Improve air circulation, mulch plants, and apply fungicides as needed." },
    'Tomato - Late blight': { 'description': "Late blight is a fast-spreading fungal disease causing large, dark, water-soaked lesions on leaves and stems.", 'treatment': "Requires prompt and thorough fungicide application. Ensure good air circulation." },
    'Tomato - Leaf Mold': { 'description': "Leaf mold causes pale green or yellowish spots on the upper leaf surface and a velvety, olive-green mold on the underside.", 'treatment': "Improve air circulation, reduce humidity, and apply fungicides. Common in greenhouses." },
    'Tomato - Septoria leaf spot': { 'description': "Septoria leaf spot is a fungal disease that creates many small, circular spots with dark borders and tan centers on leaves.", 'treatment': "Remove infected lower leaves, mulch around plants, and use fungicides." },
    'Tomato - Spider mites Two-spotted spider mite': { 'description': "Spider mites are not a disease but a pest. They cause stippling (tiny yellow dots) on leaves and fine webbing.", 'treatment': "Use miticides, insecticidal soaps, or introduce predatory mites." },
    'Tomato - Target Spot': { 'description': "Target spot is a fungal disease that creates small, water-soaked spots that enlarge into target-like lesions.", 'treatment': "Fungicide applications and good sanitation practices are key." },
    'Tomato - Tomato Yellow Leaf Curl Virus': { 'description': "This virus causes severe stunting, upward curling of leaves, and yellowing of leaf margins. It is spread by whiteflies.", 'treatment': "Control whitefly populations, use reflective mulches, and remove infected plants immediately." },
    'Tomato - Tomato mosaic virus': { 'description': "This virus causes a light and dark green mosaic pattern on leaves, along with some distortion and stunting.", 'treatment': "No cure. Focus on sanitation, use resistant varieties, and avoid handling tobacco products before touching plants." },
}

# --- Utility Functions ---
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Application UI using Streamlit native components ---
with st.container():
    st.markdown("""
    <div class="custom-header">
        <h1>🌿 Plant Disease Classification</h1>
        <p>Upload a clear image of a plant leaf, and our AI will analyze it for common diseases, providing you with instant results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state to hold the uploaded file
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    # If an image IS in the session state, show the reset button
    if st.session_state.uploaded_image is not None:
        st.markdown('<div class="card reset-button-container">', unsafe_allow_html=True)
        if st.button("Classify Another Image"):
            st.session_state.uploaded_image = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    # Otherwise, show the uploader
    else:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            st.session_state.uploaded_image = uploaded_file
            st.rerun()

    # Display "How It Works" only on the main page (when no image is uploaded)
    if st.session_state.uploaded_image is None:
        st.markdown("""
        <div class="how-it-works">
            <h2>How It Works</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="step">
                <div class="step-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 16 16"><path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z"/><path d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2h-12zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1h12z"/></svg>
                </div>
                <h3>1. Upload Image</h3>
                <p>Select a clear photo of a plant leaf from your device.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="step">
                <div class="step-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 16 16"><path d="M5 0a.5.5 0 0 1 .5.5V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2A2.5 2.5 0 0 1 14 4.5h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14A2.5 2.5 0 0 1 11.5 14v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14A2.5 2.5 0 0 1 2 11.5H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2A2.5 2.5 0 0 1 4.5 2V.5A.5.5 0 0 1 5 0zm-.5 3A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13h7a1.5 1.5 0 0 0 1.5-1.5v-7A1.5 1.5 0 0 0 11.5 3h-7zM5 6.5A1.5 1.5 0 0 1 6.5 5h3A1.5 1.5 0 0 1 11 6.5v3A1.5 1.5 0 0 1 9.5 11h-3A1.5 1.5 0 0 1 5 9.5v-3zM6.5 6a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3z"/></svg>
                </div>
                <h3>2. AI Analysis</h3>
                <p>Our system analyzes the image using a deep learning model.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="step">
                <div class="step-icon">
                     <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 16 16"><path d="M7.5 5.5a.5.5 0 0 0-1 0v.634l-.549-.317a.5.5 0 1 0-.5.866L6 7l-.549.317a.5.5 0 1 0 .5.866l.549-.317V8.5a.5.5 0 1 0 1 0v-.634l.549.317a.5.5 0 1 0 .5-.866L8 7l.549-.317a.5.5 0 1 0-.5-.866l-.549.317V5.5zm-2 4.5a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5zm0 2a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5z"/><path d="M14 14V4.5L9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2zM9.5 3A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5v2z"/></svg>
                </div>
                <h3>3. Get Results</h3>
                <p>Receive an instant diagnosis and information about the disease.</p>
            </div>
            """, unsafe_allow_html=True)

    # Display results if an image has been uploaded
    if st.session_state.uploaded_image is not None:
        image = Image.open(st.session_state.uploaded_image)
        
        display_image = image.copy()
        display_image.thumbnail((400, 400))

        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("<h3>Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(display_image, caption='Your uploaded image.')

        with res_col2:
            with st.spinner('Classifying...'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)

                predicted_class_index = np.argmax(prediction)
                predicted_class_name_raw = CLASS_NAMES[predicted_class_index]
                predicted_class_name = predicted_class_name_raw.replace('___', ' - ').replace('_', ' ')
                confidence = float(np.max(prediction))

                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                top_3_predictions = [{"class": CLASS_NAMES[i].replace('___', ' - ').replace('_', ' '), "confidence": f"{prediction[0][i]*100:.2f}%"} for i in top_3_indices]

            st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="main-prediction">
                <strong>{predicted_class_name}</strong>
                <span>(Confidence: {confidence*100:.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)

            if "healthy" in predicted_class_name_raw:
                st.markdown("""
                <div class="healthy-info">
                    <h4>✅ Plant appears to be Healthy</h4>
                    <p>No disease was detected. Continue to monitor your plant for any signs of stress or disease. Regular care, proper watering, and good nutrition are key to keeping it healthy.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                disease_info = DISEASE_INFO.get(predicted_class_name)
                if disease_info:
                    search_query = predicted_class_name.replace(" ", "+")
                    google_link = f"https://www.google.com/search?q={search_query}"
                    st.markdown(f"""
                    <div class="disease-info">
                        <h4>⚠️ Disease Details</h4>
                        <p><strong>Description:</strong> {disease_info['description']}</p>
                        <p><strong>Treatment:</strong> {disease_info['treatment']}</p>
                        <a href="{google_link}" target="_blank">Click here to learn more on Google</a>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="top-predictions">', unsafe_allow_html=True)
            st.markdown("<h3>Top 3 Predictions:</h3>", unsafe_allow_html=True)
            
            list_html = "<ul>"
            for p in top_3_predictions:
                list_html += f"<li>{p['class']} <span>{p['confidence']}</span></li>"
            list_html += "</ul>"
            
            st.markdown(list_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


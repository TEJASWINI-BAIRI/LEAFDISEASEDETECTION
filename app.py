
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from PIL import Image

app = Flask(__name__)

# Build a simple model using MobileNetV2 (no Lambda layers)
def build_simple_model():
    # Create a MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Create a new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(33, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create model
model = build_simple_model()

# Define the class labels based on your categories
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot'
]

# Dictionary of remedies for each disease
remedies = {
    'Apple___Apple_scab': 'Treatment: Apply fungicides like captan or myclobutanil. Practice good sanitation by removing infected leaves.',
    'Apple___Black_rot': 'Treatment: Prune out dead or diseased wood. Apply fungicides containing captan or thiophanate-methyl.',
    'Apple___Cedar_apple_rust': 'Treatment: Apply fungicides containing myclobutanil. Remove nearby cedar trees if possible.',
    'Apple___healthy': 'No treatment needed. Continue good gardening practices.',
    'Blueberry___healthy': 'No treatment needed. Continue good gardening practices.',
    'Cherry_(including_sour)___healthy': 'No treatment needed. Continue good gardening practices.',
    'Cherry_(including_sour)___Powdery_mildew': 'Treatment: Apply sulfur-based fungicides. Improve air circulation by pruning.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Treatment: Apply fungicides with pyraclostrobin. Rotate crops and remove debris.',
    'Corn_(maize)___Common_rust_': 'Treatment: Apply fungicides with azoxystrobin. Plant resistant varieties.',
    'Corn_(maize)___healthy': 'No treatment needed. Continue good gardening practices.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Treatment: Apply foliar fungicides. Rotate crops and use resistant varieties.',
    'Grape___Black_rot': 'Treatment: Apply fungicides with myclobutanil or captan. Remove mummified fruits.',
    'Grape___Esca_(Black_Measles)': 'Treatment: No effective fungicide available. Remove infected vines and promote vine health.',
    'Grape___healthy': 'No treatment needed. Continue good gardening practices.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Treatment: Apply copper-based fungicides. Remove infected leaves.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Treatment: No cure available. Remove infected trees and control psyllid vectors.',
    'Peach___Bacterial_spot': 'Treatment: Apply copper-based bactericides. Prune during dry weather.',
    'Peach___healthy': 'No treatment needed. Continue good gardening practices.',
    'Pepper,_bell___Bacterial_spot': 'Treatment: Apply copper-based bactericides. Rotate crops.',
    'Pepper,_bell___healthy': 'No treatment needed. Continue good gardening practices.',
    'Potato___Early_blight': 'Treatment: Apply fungicides with chlorothalonil. Maintain good soil fertility.',
    'Potato___healthy': 'No treatment needed. Continue good gardening practices.',
    'Potato___Late_blight': 'Treatment: Apply fungicides with mancozeb or chlorothalonil. Remove infected plants.',
    'Raspberry___healthy': 'No treatment needed. Continue good gardening practices.',
    'Soybean___healthy': 'No treatment needed. Continue good gardening practices.',
    'Squash___Powdery_mildew': 'Treatment: Apply sulfur-based fungicides. Space plants for good air circulation.',
    'Strawberry___healthy': 'No treatment needed. Continue good gardening practices.',
    'Strawberry___Leaf_scorch': 'Treatment: Apply fungicides with captan. Remove infected leaves.',
    'Tomato___Bacterial_spot': 'Treatment: Apply copper-based bactericides. Rotate crops and avoid overhead irrigation.',
    'Tomato___Early_blight': 'Treatment: Apply fungicides with chlorothalonil. Remove lower infected leaves.',
    'Tomato___Late_blight': 'Treatment: Apply fungicides with chlorothalonil or mancozeb. Remove infected plants.',
    'Tomato___Leaf_Mold': 'Treatment: Improve air circulation and reduce humidity. Apply fungicides with chlorothalonil.',
    'Tomato___Septoria_leaf_spot': 'Treatment: Apply fungicides with chlorothalonil. Remove infected leaves and practice crop rotation.'
}

def preprocess_image(image):
    # Resize to the model's expected input size
    img = cv2.resize(image, (224, 224))
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Preprocess the image for MobileNetV2
    img = img / 127.5 - 1.0  # Normalize to [-1, 1]
    return np.expand_dims(img, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read and preprocess the image
        img = Image.open(file.stream)
        img = np.array(img)
        
        # Preprocess the image
        processed_img = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(processed_img)[0]
        
        # Get top prediction
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = float(prediction[predicted_class_index])
        
        # Get remedy for the predicted disease
        remedy = remedies.get(predicted_class, "No specific remedy information available.")
        
        # Return prediction results
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence * 100,  # Convert to percentage
            'remedy': remedy
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate
# from tensorflow.keras.applications import ResNet50
# from keras_cv.models import VisionTransformer
# from PIL import Image

# app = Flask(__name__)

# def build_vit_resnet_model():
#     # Load pre-trained ResNet50 model
#     resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     resnet_features = GlobalAveragePooling2D()(resnet_base.output)
    
#     # Load pre-trained ViT model
#     vit_base = VisionTransformer.from_preset("vit_small_imagenet")
#     vit_features = GlobalAveragePooling2D()(vit_base(resnet_base.input))
    
#     # Extract bottleneck features
#     combined_features = Concatenate()([resnet_features, vit_features])
    
#     # Fully connected layers
#     output = Dense(128, activation='relu')(combined_features)
#     output = Dense(33, activation='softmax')(output)
    
#     # Create model
#     model = Model(inputs=resnet_base.input, outputs=output)
    
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model

# # Load model
# model = build_vit_resnet_model()

# # Define class labels
# class_labels = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#     'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#     'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
#     'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
#     'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
#     'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 
#     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
#     'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot'
# ]

# # Remedies dictionary
# remedies = {
#     'Apple___Apple_scab': 'Apply fungicides like captan or myclobutanil. Remove infected leaves.',
#     'Apple___Black_rot': 'Prune dead wood. Use fungicides containing captan or thiophanate-methyl.',
#     'Apple___Cedar_apple_rust': 'Apply fungicides with myclobutanil. Remove nearby cedar trees.',
#     'Apple___healthy': 'No treatment needed. Maintain good gardening practices.',
#     'Blueberry___healthy': 'No treatment needed.',
#     'Cherry_(including_sour)___healthy': 'No treatment needed.',
#     'Cherry_(including_sour)___Powdery_mildew': 'Use sulfur-based fungicides. Improve air circulation.',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply pyraclostrobin fungicides. Rotate crops.',
#     'Corn_(maize)___Common_rust_': 'Use azoxystrobin fungicides. Plant resistant varieties.',
#     'Corn_(maize)___healthy': 'No treatment needed.',
#     'Corn_(maize)___Northern_Leaf_Blight': 'Apply foliar fungicides. Rotate crops.',
#     'Grape___Black_rot': 'Use myclobutanil or captan fungicides. Remove mummified fruits.',
#     'Grape___Esca_(Black_Measles)': 'No cure. Remove infected vines.',
#     'Grape___healthy': 'No treatment needed.',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use copper-based fungicides. Remove infected leaves.',
#     'Orange___Haunglongbing_(Citrus_greening)': 'No cure. Remove infected trees.',
#     'Peach___Bacterial_spot': 'Use copper-based bactericides. Prune during dry weather.',
#     'Peach___healthy': 'No treatment needed.',
#     'Pepper,_bell___Bacterial_spot': 'Use copper-based bactericides. Rotate crops.',
#     'Pepper,_bell___healthy': 'No treatment needed.',
#     'Potato___Early_blight': 'Apply chlorothalonil fungicides. Maintain soil fertility.',
#     'Potato___healthy': 'No treatment needed.',
#     'Potato___Late_blight': 'Use mancozeb or chlorothalonil fungicides. Remove infected plants.',
#     'Raspberry___healthy': 'No treatment needed.',
#     'Soybean___healthy': 'No treatment needed.',
#     'Squash___Powdery_mildew': 'Apply sulfur-based fungicides. Space plants properly.',
#     'Strawberry___healthy': 'No treatment needed.',
#     'Strawberry___Leaf_scorch': 'Use captan fungicides. Remove infected leaves.',
#     'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Avoid overhead watering.',
#     'Tomato___Early_blight': 'Use chlorothalonil fungicides. Remove lower infected leaves.',
#     'Tomato___Late_blight': 'Use chlorothalonil or mancozeb fungicides. Remove infected plants.',
#     'Tomato___Leaf_Mold': 'Improve air circulation. Use chlorothalonil fungicides.',
#     'Tomato___Septoria_leaf_spot': 'Apply chlorothalonil fungicides. Remove infected leaves.'
# }

# def preprocess_image(image):
#     img = cv2.resize(image, (224, 224))
#     img = img / 127.5 - 1.0
#     return np.expand_dims(img, axis=0)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     try:
#         img = Image.open(file.stream)
#         img = np.array(img)
#         processed_img = preprocess_image(img)
        
#         prediction = model.predict(processed_img)[0]
#         predicted_class_index = np.argmax(prediction)
#         predicted_class = class_labels[predicted_class_index]
#         confidence = float(prediction[predicted_class_index])
#         remedy = remedies.get(predicted_class, 'No specific remedy available.')
        
#         return jsonify({'prediction': predicted_class, 'confidence': confidence * 100, 'remedy': remedy})
    
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

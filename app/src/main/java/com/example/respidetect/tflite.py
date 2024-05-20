import numpy as np
import librosa

# Define the list of classes
classes = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="app/src/main/assets/quantized_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the audio data
def preprocess_audio(audio_file, sample_rate):
    audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
    # Perform any necessary preprocessing (e.g., extracting MFCCs)
    # Make sure the preprocessed data matches the input format expected by the model
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=52).T,axis = 0)
    mfccs = mfccs.reshape(1, 52, 1)

    # Convert MFCCs to bytes
    audio_data = mfccs.tobytes()

    return audio_data

# Run inference on the preprocessed audio data using the loaded model
def predict_disease(audio_data):
    # Perform inference using the loaded model
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the corresponding disease label based on the index
    predicted_disease = classes[predicted_class_index]

    return predicted_disease

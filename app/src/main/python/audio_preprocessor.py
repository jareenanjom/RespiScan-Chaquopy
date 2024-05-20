import tflite_runtime.interpreter as tflite
import numpy as np
import librosa

# Define the list of classes
classes = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]

# Load your TFLite model
interpreter = tflite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(audio_data, sample_rate):
    audio = np.array(audio_data, dtype=np.float32)
    # Perform any necessary preprocessing (e.g., extracting MFCCs)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=52).T, axis=0)
    mfccs = mfccs.reshape(1, 52, 1)
    return mfccs

def predict_disease(audio_data):
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions)
    predicted_disease = classes[predicted_class_index]
    return predicted_disease

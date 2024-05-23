package com.example.respidetect;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import com.example.respidetect.audio.features.MFCC;
import com.example.respidetect.noiseclassifier.Recognition;

import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static final String MODEL_PATH = "quantized_model.tflite";
    private static final int SAMPLE_RATE = 44100;
    private static final int RECORDING_LENGTH = 20; // in seconds
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    private boolean permissionToRecordAccepted = false;
    private String[] permissions = {Manifest.permission.RECORD_AUDIO};
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private Interpreter interpreter;
    private TextView textViewOutput;
    private TextView textViewSpec;
    private TextView textViewTrachea;
    private TextView textViewAL;
    private TextView textViewAR;
    private TextView textViewPL;
    private TextView textViewPR;
    private TextView textViewLL;
    private TextView textViewLR;
    private TextView[] textViews;
    private int currentTextViewIndex = 0;

    private short[] audioData;
    private int bufferSize;
    private Thread recordingThread;
    private List<String> textViewNames = Arrays.asList("Trachea", "Anterior Left", "Anterior Right", "Posterior Left", "Posterior Right", "Lateral Left", "Lateral Right");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textViewOutput = findViewById(R.id.textViewOutput);
        textViewSpec = findViewById(R.id.textViewSpec);
        textViewTrachea = findViewById(R.id.textViewTrachea);
        textViewAL = findViewById(R.id.textViewAL);
        textViewAR = findViewById(R.id.textViewAR);
        textViewPL = findViewById(R.id.textViewPL);
        textViewPR = findViewById(R.id.textViewPR);
        textViewLL = findViewById(R.id.textViewLL);
        textViewLR = findViewById(R.id.textViewLR);
        textViews = new TextView[]{textViewTrachea, textViewAL, textViewAR, textViewPL, textViewPR, textViewLL, textViewLR};

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);
        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
    }

    private int numShortsRead = 0;
    public void onStartRecording(View view) {
        if (permissionToRecordAccepted && !isRecording) {
            Log.d("AudioRecording", "Recording is about to start...");

            bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                // TODO: Consider calling
                //    ActivityCompat#requestPermissions
                // here to request the missing permissions, and then overriding
                //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                //                                          int[] grantResults)
                // to handle the case where the user grants the permission. See the documentation
                // for ActivityCompat#requestPermissions for more details.
                return;
            }
            audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);

            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e("AudioRecording", "AudioRecord initialization failed!");
                return;
            }

            audioData = new short[bufferSize];
            audioRecord.startRecording();
            isRecording = true;

            Log.d("AudioRecording", "Recording started successfully.");

            recordingThread = new Thread(() -> {
                while (isRecording) {
                    int read = audioRecord.read(audioData, 0, bufferSize);
                    if (read < 0) {
                        Log.e("AudioRecording", "Failed to read audio data!");
                    } else {
                        numShortsRead += read;
                    }
                }
            }, "Audio Recording Thread");
            recordingThread.start();
            numShortsRead = 0; // Reset the count
        }
    }


    public void onStopRecording(View view) throws IOException {
        if (isRecording) {
            isRecording = false;
            audioRecord.stop();

            Log.d("AudioRecording", "Recording stopped.");

            double recordedSeconds = (double) numShortsRead /  SAMPLE_RATE;
            Log.d("Recording Duration", "Recorded " + recordedSeconds + " seconds");
            // Start processing in a background thread
            new AudioProcessingTask().execute(audioData);
        }
    }



    private String preprocessAudio(short[] audioData) throws IOException {
        try {
            Python py = Python.getInstance();
            PyObject audioProcessor = py.getModule("audio_preprocessor");

            int sampleRate = SAMPLE_RATE;
            float[] audioDataFloat = new float[audioData.length];
            for (int i = 0; i < audioData.length; i++) {
                audioDataFloat[i] = audioData[i];
            }

            PyObject mfccs = audioProcessor.callAttr("preprocess_audio", audioDataFloat, sampleRate);
            PyObject prediction = audioProcessor.callAttr("predict_disease", mfccs);

            return prediction.toString();
        } catch (Exception ex) {
            Log.e("preprocessAudio", "Exception caught: " + ex.getMessage(), ex);
            throw new IOException("Error processing audio data", ex);
        }
    }


    private void updateUIWithPrediction(String prediction) {
        Map<String, Integer> predictionCounts = new HashMap<>();
        String textViewName = textViewNames.get(currentTextViewIndex);
        textViews[currentTextViewIndex].setText(textViewName + ": " + prediction);

        // Increment prediction count
        predictionCounts.put(prediction, predictionCounts.getOrDefault(prediction, 0) + 1);
        currentTextViewIndex++;
        if (currentTextViewIndex >= textViews.length) {
            // Determine majority prediction and update UI...
            String majorityPrediction = "";
            int maxCount = 0;
            for (Map.Entry<String, Integer> entry : predictionCounts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    majorityPrediction = entry.getKey();
                    maxCount = entry.getValue();
                }
            }

            // Update the output TextView with the majority prediction
            textViewOutput.setText("Predicted disease: " + majorityPrediction);

            // Reset index if all TextViews are updated
            currentTextViewIndex = 0;
            predictionCounts.clear(); //
        }
    }
    protected String loadModelAndMakePredictions(float[] meanMFCCValues) throws IOException {
        String predictedResult = "unknown";

        // Load the TFLite model
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(this, MODEL_PATH);
        Interpreter tflite;

        // Configure the interpreter options
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(2);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // Obtain input and output tensor information
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Transform the MFCC buffer into the required tensor shape
        TensorBuffer inBuffer = TensorBuffer.createDynamic(imageDataType);
        inBuffer.loadArray(meanMFCCValues, imageShape);

        // Define output tensor
        TensorBuffer outputTensorBuffer
                = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Run the predictions
        tflite.run(inBuffer.getBuffer(), outputTensorBuffer.getBuffer());

        // Transform the probability predictions into label values
        String ASSOCIATED_AXIS_LABELS = "labels.txt";
        List<String> associatedAxisLabels = null;
        try {
            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        // Tensor processor for processing the probability values and sorting them
        TensorProcessor probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        if (associatedAxisLabels != null) {
            // Map of labels and their corresponding probability
            TensorLabel labels = new TensorLabel(associatedAxisLabels, probabilityProcessor.process(outputTensorBuffer));

            // Retrieve the top K probability values
            Map<String, Float> floatMap = labels.getMapWithFloatValue();

            // Retrieve the top 1 prediction
            List<Recognition> resultPrediction = getTopKProbability(floatMap);

            // Get the predicted value
            predictedResult = getPredictedValue(resultPrediction);
        }
        return predictedResult;
    }

    private String getPredictedValue(List<Recognition> predictedList) {
        Recognition top1PredictedValue = predictedList != null ? predictedList.get(0) : null;
        return top1PredictedValue != null ? top1PredictedValue.getTitle() : null;
    }

    protected List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications
        int MAX_RESULTS = 1;
        PriorityQueue<Recognition> pq = new PriorityQueue<>(MAX_RESULTS, (lhs, rhs) -> Float.compare(rhs.getConfidence(), lhs.getConfidence()));
        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition(entry.getKey(), entry.getKey(), entry.getValue()));
        }
        List<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; i++) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    private class AudioProcessingTask extends AsyncTask<short[], Void, String> {
        @Override
        protected String doInBackground(short[]... audioData) {
            try {
                return preprocessAudio(audioData[0]);
            } catch (IOException e) {
                Log.e("AudioProcessingTask", "Error processing audio", e);
                return null;
            }
        }

        @Override
        protected void onPostExecute(String prediction) {
            if (prediction != null) {
                updateUIWithPrediction(prediction);
            } else {
                Toast.makeText(MainActivity.this, "Failed to process audio", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_RECORD_AUDIO_PERMISSION:
                permissionToRecordAccepted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
                if (permissionToRecordAccepted) {
                    // Initialize audio record if permission is granted
                    initializeAudioRecord();
                } else {
                    // Handle the case where permissions are not granted
                    Toast.makeText(this, "Recording permission is required to use this app.", Toast.LENGTH_SHORT).show();
                }
                break;
        }
    }

    private void initializeAudioRecord() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            // Permission not granted, request it
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO_PERMISSION);
            return; // Return without initializing audioRecord
        }

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (isRecording) {
            stopRecording();
        }
    }

    private void stopRecording() {
        isRecording = false;
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
        }
        if (recordingThread != null && recordingThread.isAlive()) {
            recordingThread.interrupt();
        }
    }
}

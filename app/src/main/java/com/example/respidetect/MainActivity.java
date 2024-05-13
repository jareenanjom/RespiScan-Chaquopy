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

import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.math.RoundingMode;

//import org.jtransforms.fft.DoubleFFT_1D;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;


import com.example.respidetect.audio.features.MFCC;
import com.example.respidetect.noiseclassifier.Recognition;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static final String MODEL_PATH = "resp_model.tflite";
    private static final int SAMPLE_RATE = 44100;
    private static final int RECORDING_LENGTH = 20; // in seconds
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int SAMPLE_DURATION = SAMPLE_RATE * RECORDING_LENGTH;

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

    private short[] audioData;
    private Thread recordingThread = null;
    private TextView[] textViews;
    private int currentTextViewIndex = 0;


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
    }


    public void onStartRecording(View view) {
        if (permissionToRecordAccepted && !isRecording) {
            Log.d("AudioRecording", "Recording is about to start...");

            isRecording = true;
            audioData = null;
            audioRecord.startRecording();
            Log.d("AudioRecording", "Recording started successfully.");


        }
    }


    private String preprocessAudio(short[] audioData) throws IOException {
        List<String> predictions = new ArrayList<>();

        int read = audioRecord.read(audioData, 0, audioData.length);
        int mChannels = audioRecord.getChannelCount();
        int mSampleRate = audioRecord.getSampleRate();
        int mNumFrames = read / (audioRecord.getAudioFormat() == AudioFormat.ENCODING_PCM_16BIT ? 2 : 1);
        double[][] buffer = new double[mChannels][mNumFrames];
        String prediction = "Unknown";
        Log.d("AudioProcessing", "Starting audio preprocessing...");


        // Perform audio processing and feature extraction here

        DecimalFormat df = new DecimalFormat("#.#####");
        df.setRoundingMode(RoundingMode.CEILING);
        double[] meanBuffer = new double[mNumFrames];
        for (int q = 0; q < mNumFrames; q++) {
            double frameVal = 0.0;
            for (int p = 0; p < mChannels; p++) {
                frameVal += buffer[p][q];
            }
            meanBuffer[q] = Double.parseDouble(df.format(frameVal / mChannels));
        }

        MFCC mfccConvert = new MFCC();
        mfccConvert.setSampleRate(mSampleRate);
        int nMFCC = 52;
        mfccConvert.setN_mfcc(nMFCC);
        float[] mfccInput = mfccConvert.process(meanBuffer);
        int nFFT = mfccInput.length / nMFCC;
        double[][] mfccValues = new double[nMFCC][nFFT];


        //loop to convert the mfcc values into multi-dimensional array
        for (int i = 0; i < nFFT; i++) {
            int indexCounter = i * nMFCC;
            int rowIndexValue = i % nFFT;
            for (int j = 0; j < nMFCC; j++) {
                mfccValues[j][rowIndexValue] = mfccInput[indexCounter];
                indexCounter++;
            }
        }

        //mean mfcc values
        float[] meanMFCCValues = new float[nMFCC];
        for (int p = 0; p < nMFCC; p++) {
            double fftValAcrossRow = 0.0;
            for (int q = 0; q < nFFT; q++) {
                fftValAcrossRow = fftValAcrossRow + mfccValues[p][q];
            }
            double fftMeanValAcrossRow = fftValAcrossRow / nFFT;
            meanMFCCValues[p] = (float) fftMeanValAcrossRow;
        }

        prediction = loadModelAndMakePredictions(meanMFCCValues);
        Log.d("AudioProcessing", "Audio preprocessing completed.");

        return prediction;

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
        ByteBuffer inpBuffer = inBuffer.getBuffer();
        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
        // Run the predictions
        tflite.run(inpBuffer, outputTensorBuffer.getBuffer());

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


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_RECORD_AUDIO_PERMISSION:
                permissionToRecordAccepted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
                if (permissionToRecordAccepted) {
                    initializeAudioRecord();
                } else {
                    // Handle the case where permissions are not granted
                    Toast.makeText(this, "Recording permission is required to use this app.", Toast.LENGTH_SHORT).show();
                }
                break;
        }
    }

    private void initializeAudioRecord() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            // Permission not granted, request it
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO_PERMISSION);
            return; // Return without initializing audioRecord
        }

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);
    }

    private class AudioProcessingTask extends AsyncTask<Void, Void, String>  {

        private TextView[] textViews;

        public AudioProcessingTask(TextView[] textViews) {
            this.textViews = textViews;
        }

        @Override
        protected String doInBackground(Void... voids) {
            // Perform audio processing here
            try {
                return preprocessAudio(audioData); // Return only one prediction
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        private List<String> textViewNames = Arrays.asList("Trachea", "Anterior Left", "Anterior Right", "Posterior Left", "Posterior Right", "Lateral Left", "Lateral Right");

        @Override
        public void onPostExecute(String prediction) {
            String textViewName = textViewNames.get(currentTextViewIndex);

            textViews[currentTextViewIndex].setText(textViewName + ": " + prediction);

            // Move to the next TextView for the next recording
            currentTextViewIndex++;
            if (currentTextViewIndex >= textViews.length) {
                currentTextViewIndex = 0; // Reset index if all TextViews are updated
            }
        }
        }



        public void onStopRecording(View view) throws IOException {
            List<String> textViewNames = Arrays.asList("Trachea", "Anterior Left", "Anterior Right", "Posterior Left", "Posterior Right", "Lateral Left", "Lateral Right");
            Map<String, Integer> predictionCounts = new HashMap<>();

            if (isRecording) {
                isRecording = false;
                audioRecord.stop();
                double recordedSeconds = audioRecord.getBufferSizeInFrames() / SAMPLE_RATE * 0.001;

                audioData = new short[audioRecord.getBufferSizeInFrames() * audioRecord.getChannelCount()];
                Log.d("Recording Duration", "Recorded " + recordedSeconds + " seconds");

                // Update the current TextView with the prediction
                String prediction = preprocessAudio(audioData);
                String textViewName = textViewNames.get(currentTextViewIndex);

                textViews[currentTextViewIndex].setText(textViewName + ": " + prediction);

                // Increment prediction count
                predictionCounts.put(prediction, predictionCounts.getOrDefault(prediction, 0) + 1);

                // Move to the next TextView for the next recording
                currentTextViewIndex++;
                if (currentTextViewIndex >= textViews.length) {
                    // Determine majority prediction
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
                    predictionCounts.clear(); // Reset prediction counts
                }
            }
        }


}


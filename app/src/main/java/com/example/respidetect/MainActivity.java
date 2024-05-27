package com.example.respidetect;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
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

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static final int REQUEST_READ_EXTERNAL_STORAGE_PERMISSION = 201;
    private static final int REQUEST_AUDIO_FILE = 202;
    private static final String MODEL_PATH = "quantized_model.tflite";
    private static final int SAMPLE_RATE = 44100;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    private boolean permissionToRecordAccepted = false;
    private boolean permissionToReadStorageAccepted = false;
    private String[] permissions = {Manifest.permission.RECORD_AUDIO};
    private String[] storagePermissions = {Manifest.permission.READ_EXTERNAL_STORAGE};
    private boolean isRecording = false;
    private int bufferSize;
    private Thread recordingThread;

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
    private List<String> predictionsList = new ArrayList<>();

    private short[] audioData;
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
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        ActivityCompat.requestPermissions(this, storagePermissions, REQUEST_READ_EXTERNAL_STORAGE_PERMISSION);
    }

    public void onStartRecording(View view) {
        if (permissionToRecordAccepted && !isRecording) {
            startNewRecording();
        }
    }

    public void onStopRecording(View view) throws IOException, InterruptedException {
        if (isRecording) {
            stopCurrentRecording();
        }
    }

    private void startNewRecording() {
        bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        AudioRecord audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);
        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e("AudioRecording", "AudioRecord initialization failed!");
            return;
        }

        audioData = new short[bufferSize];
        audioRecord.startRecording();
        isRecording = true;
        Log.d("AudioRecording", "Recording started successfully.");

        recordingThread = new Thread(() -> {
            int read;
            while (isRecording) {
                read = audioRecord.read(audioData, 0, bufferSize);
                if (read < 0) {
                    Log.e("AudioRecording", "Failed to read audio data!");
                }
            }
            audioRecord.stop();
            audioRecord.release();
        }, "Audio Recording Thread");
        recordingThread.start();
    }

    private void stopCurrentRecording() throws IOException, InterruptedException {
        isRecording = false;
        recordingThread.join();

        // Start processing in a background thread
        new AudioProcessingTask().execute(audioData);
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
        if (currentTextViewIndex < textViews.length) {
            String textViewName = textViewNames.get(currentTextViewIndex);
            textViews[currentTextViewIndex].setText(textViewName + ": " + prediction);
            predictionsList.add(prediction);
            currentTextViewIndex++;
            if (currentTextViewIndex == textViews.length) {
                calculateAndDisplayMajorityVote();
            }
        } else {
            Toast.makeText(this, "All TextViews have been updated.", Toast.LENGTH_SHORT).show();
            currentTextViewIndex = 0;
        }
    }

    private void calculateAndDisplayMajorityVote() {
        Map<String, Integer> predictionCountMap = new HashMap<>();
        for (String prediction : predictionsList) {
            predictionCountMap.put(prediction, predictionCountMap.getOrDefault(prediction, 0) + 1);
        }
        String majorityPrediction = null;
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : predictionCountMap.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityPrediction = entry.getKey();
            }
        }
        textViewOutput.setText("Majority Vote: " + majorityPrediction);
        predictionsList.clear();
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
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            permissionToRecordAccepted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            if (!permissionToRecordAccepted) {
                Toast.makeText(this, "Recording permission is required to use this app.", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_READ_EXTERNAL_STORAGE_PERMISSION) {
            permissionToReadStorageAccepted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            if (!permissionToReadStorageAccepted) {
                Toast.makeText(this, "Storage permission is required to select audio files.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    public void onSelectAudioFile(View view) {
        if (permissionToReadStorageAccepted) {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("audio/*");
            startActivityForResult(intent, REQUEST_AUDIO_FILE);
        } else {
            ActivityCompat.requestPermissions(this, storagePermissions, REQUEST_READ_EXTERNAL_STORAGE_PERMISSION);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_AUDIO_FILE && resultCode == RESULT_OK && data != null) {
            Uri audioUri = data.getData();
            new AudioProcessingTaskForFile().execute(audioUri);
        }
    }

    private class AudioProcessingTaskForFile extends AsyncTask<Uri, Void, String> {
        @Override
        protected String doInBackground(Uri... uris) {
            try {
                InputStream inputStream = getContentResolver().openInputStream(uris[0]);
                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    byteArrayOutputStream.write(buffer, 0, bytesRead);
                }
                byte[] audioBytes = byteArrayOutputStream.toByteArray();
                short[] audioData = new short[audioBytes.length / 2];
                ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(audioData);
                inputStream.close();
                byteArrayOutputStream.close();
                return preprocessAudio(audioData);
            } catch (IOException e) {
                Log.e("AudioProcessingTaskForFile", "Error processing audio file", e);
                return null;
            }
        }

        @Override
        protected void onPostExecute(String prediction) {
            if (prediction != null) {
                updateUIWithPrediction(prediction);
            } else {
                Toast.makeText(MainActivity.this, "Failed to process audio file", Toast.LENGTH_SHORT).show();
            }
        }
    }
}

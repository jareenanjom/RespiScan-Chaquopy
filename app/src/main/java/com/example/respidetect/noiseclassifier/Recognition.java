package com.example.respidetect.noiseclassifier;


public class Recognition {

    private String id;
    private String title;
    private float confidence;

    public Recognition(String id, String title, float confidence) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public float getConfidence() {
        return confidence;
    }
}

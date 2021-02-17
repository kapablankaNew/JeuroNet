package org.kapablankaNew.JeuroNet;

public class DataSetException extends Exception {
    public DataSetException(String message) {
        super(message);
    }

    @Override
    public String toString() {
        String result = super.toString();
        return "Error in dataset: " + result;
    }
}
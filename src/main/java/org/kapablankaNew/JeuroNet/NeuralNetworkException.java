package org.kapablankaNew.JeuroNet;

public class NeuralNetworkException extends Exception {
    public NeuralNetworkException(String Message) {
        super(Message);
    }

    @Override
    public String toString() {
        String result = super.toString();
        return "Error in neural network: " + result;
    }
}

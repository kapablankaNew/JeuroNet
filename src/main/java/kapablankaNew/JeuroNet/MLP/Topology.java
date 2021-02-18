package kapablankaNew.JeuroNet.MLP;

import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Topology implements Serializable {
    //number of inputs in neural network (in first layer)
    private final int inputCount;
    //number of outputs of neural network (ih last layer)
    private final int outputCount;
    //number of neurons in several hidden layer
    private final List<Integer> hiddenLayers;

    private final double learningRate;

    public Topology(int inputCount, int outputCount, @NonNull int[] layers, double learningRate) throws TopologyException {
        if (inputCount <= 0) {
            throw new TopologyException("Number of inputs must be greater than 0");
        }
        if (outputCount <= 0) {
            throw new TopologyException("Number of outputs must be greater than 0");
        }
        if (learningRate <= 0) {
            throw new TopologyException("Learning rate must be greater than 0");
        }
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.learningRate = learningRate;
        hiddenLayers = new ArrayList<>();
        for (int layer : layers) {
            hiddenLayers.add(layer);
        }
    }

    public Topology(int inputCount, int outputCount, int layer, double learningRate) throws TopologyException {
        this(inputCount, outputCount, new int[]{layer}, learningRate);
    }

    public Topology(int inputCount, int outputCount, double learningRate) throws TopologyException {
        this(inputCount, outputCount, new int[]{outputCount}, learningRate);
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getInputCount() {
        return inputCount;
    }

    public int getOutputCount() {
        return outputCount;
    }

    public List<Integer> getHiddenLayers() {
        return hiddenLayers;
    }

    public int getCountOfNeuronsInLayer(int index) {
        return hiddenLayers.get(index);
    }
}

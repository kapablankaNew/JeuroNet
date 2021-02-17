package org.kapablankaNew.JeuroNet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron implements Serializable {
    //list of weights of the synapses between current neuron and previous layer
    private final List<Double> weights;

    private final NeuronType neuronType;
    //list for storage last inputs signals
    private final List<Double> inputs;
    //variable for storage last calculated value of output signal
    private double output;
    //variable for storage value of error
    private double delta;

    public Neuron(int inputCount, NeuronType type) {
        neuronType = type;
        weights = new ArrayList<>();
        inputs = new ArrayList<>();
        //any input neuron have only 1 input with weight = 1
        //this neuron only accepts input data
        //and translate this data to all the neurons of the first hidden layer
        if (neuronType ==  NeuronType.Input) {
            weights.add(1.0);
            inputs.add(1.0);
        } else {
            //another neurons have several inputs with random weights
            initWeightsRandomValues(inputCount);
        }
    }

    public Neuron(int inputCount) {
        this(inputCount, NeuronType.Normal);
    }

    public double getDelta() {
        return delta;
    }

    public List<Double> getInputs() {
        return inputs;
    }

    private void initWeightsRandomValues(int inputCount) {
        Random random = new Random();
        for (int i = 0; i < inputCount; i++) {
            //Random.nextDouble() return value between 0 and 1
            //using simple conversion, getting result between -1 and 1
            weights.add(random.nextDouble() * 2 - 1.0);
            inputs.add(0.0);
        }
    }

    public List<Double> getWeights() {
        return weights;
    }

    public double getResult() {
        return output;
    }

    //we create a simple neural network, signal goes from left to right
    //if the are several loop of going signal - it is recurrent neural network
    public void feedForward(List<Double> inputs) {
        //saving input signals
        this.inputs.clear();
        this.inputs.addAll(inputs);

        //calculating sum of input signals
        double sum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs.get(i) * weights.get(i);
        }
        //input neuron doesn't have activation function
        //they only translate data to all the neurons of the first hidden layer
        if (neuronType == NeuronType.Input) {
            output = sum;
            return;
        }
        //another neurons have sigmoid as activation function
        output = sigmoid(sum);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    //calculating error for learning using back propagation
    public void calculateError(double expectedOutput) {
        //for output neurons error calculating using expected value of output
        //for neuron i:
        //error = OUTi*(1 - OUTi)*(EXPECTED_OUTi - OUTi)
        //OUTi*(1 - OUTi) = d(sigmoid(x))/dx
        if (neuronType == NeuronType.Output) {
            delta = output * (1 - output) * (expectedOutput - output);
        }
    }

    public void calculateError(Layer nextLayer, int currentNeuronNumber) {
        //input neurons aren't learn
        if (neuronType == NeuronType.Input) {
            return;
        }
        //for calculating the error for hidden neurons use the values of errors of next layer
        //error_i = OUTi*(1 - OUTi)*(sum_children)
        //sum_children = summary (weights(i->j)*error_j), j - all neurons of the next layer
        double sum = 0;
        //getting all neurons of the next layer
        List<Neuron> neuronList = nextLayer.getNeurons();
        for (Neuron neuron : neuronList) {
            //getting neuron
            //adding weight*delta to the summary
            sum += neuron.getDelta() * neuron.getWeights().get(currentNeuronNumber);
        }
        //calculating error
        delta = output * (1 - output) * sum;
    }

    public void learnBackPropagation(double learningRate) {
        //input neurons aren't learning
        if (neuronType == NeuronType.Input) {
            return;
        }
        //another neurons updating weights
        //newW_i = oldW_i + LR*error*input_i
        for (int i = 0; i < weights.size(); i++) {
            double w = weights.get(i) + learningRate * delta * inputs.get(i);
            weights.set(i, w);
        }
    }

    @Override
    public String toString() {
        return String.valueOf(output);
    }
}

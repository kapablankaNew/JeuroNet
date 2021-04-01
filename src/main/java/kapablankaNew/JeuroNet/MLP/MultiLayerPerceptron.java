package kapablankaNew.JeuroNet.MLP;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import lombok.NonNull;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import kapablankaNew.JeuroNet.*;

public class MultiLayerPerceptron implements Serializable {
    private final List<Layer> layers;

    private final Topology topology;

    private DataSet lastDataSet;

    private final boolean scalingData;

    public MultiLayerPerceptron(Topology topology, boolean scalingData) {
        this.topology = topology;
        this.scalingData = scalingData;
        lastDataSet = null;
        layers = new ArrayList<>();
        createInputLayer();
        createHiddenLayers();
        createOutputLayer();
    }

    public MultiLayerPerceptron(Topology topology) {
        this(topology, false);
    }

    private void createInputLayer() {
        List<Neuron> inputNeurons = new ArrayList<>();
        //filling the layer with neurons
        for (int i = 0; i < topology.getInputCount(); i++) {
            //input neuron always have 1 input
            Neuron neuron = new Neuron(1, NeuronType.Input, ActivationFunction.LINEAR);
            inputNeurons.add(neuron);
        }
        //creating layer ad adding this in list of layers
        Layer inputLayer = new Layer(inputNeurons, NeuronType.Input, ActivationFunction.LINEAR);
        layers.add(inputLayer);
    }

    private void createHiddenLayers() {
        for (int i = 0; i < topology.getHiddenLayersInfos().size(); i++) {
            List<Neuron> hiddenNeurons = new ArrayList<>();
            //getting last layer (its size is the number of inputs of each neuron in this layer layer)
            Layer lastLayer = layers.get(layers.size() - 1);
            //filling the layer with neurons
            for (int j = 0; j < topology.getCountOfNeuronsInLayer(i); j++) {
                Neuron neuron = new Neuron(lastLayer.getCount(), NeuronType.Normal,
                        topology.getHiddenLayersInfos().get(i).getActivationFunction());
                hiddenNeurons.add(neuron);
            }
            //creating layer ad adding this in list of layers
            Layer hiddenLayer = new Layer(hiddenNeurons, NeuronType.Normal,
                    topology.getHiddenLayersInfos().get(i).getActivationFunction());
            layers.add(hiddenLayer);
        }
    }

    private void createOutputLayer() {
        List<Neuron> outputNeurons = new ArrayList<>();
        //getting last layer (its size is the number of inputs of each neuron in the output layer)
        Layer lastLayer = layers.get(layers.size() - 1);
        //filling the layer with neurons
        for (int i = 0; i < topology.getOutputCount(); i++) {
            Neuron neuron = new Neuron(lastLayer.getCount(), NeuronType.Output,
                    topology.getOutputLayerInfo().getActivationFunction());
            outputNeurons.add(neuron);
        }
        //creating layer ad adding this in list of layers
        Layer outputLayer = new Layer(outputNeurons, NeuronType.Output,
                topology.getOutputLayerInfo().getActivationFunction());
        layers.add(outputLayer);
    }

    @NonNull
    public List<Double> predict(List<Double> inputSignals) throws MultiLayerPerceptronException, DataSetException {
        if (lastDataSet == null) {
            throw new MultiLayerPerceptronException("Neural network isn't trained");
        }
        if (inputSignals.size() != topology.getInputCount()) {
            throw new MultiLayerPerceptronException("The number of input signals not equals " +
                    "to the number of inputs of neural network!");
        }
        if (scalingData) {
            //scaling input signals
            lastDataSet.scaleEntry(inputSignals);
        }
        //for feed forward sending signals to input neurons
        sendSignalsToInputNeurons(inputSignals);
        //after this go through all the other layers
        feedForwardAllLayersAfterInput();
        //return list of outputs
        return layers.get(layers.size() - 1).getOutputSignals();
    }

    private void predictLearning(List<Double> inputSignals) {
        //for feed forward sending signals to input neurons
        sendSignalsToInputNeurons(inputSignals);
        //after this go through all the other layers
        feedForwardAllLayersAfterInput();
    }

    private void sendSignalsToInputNeurons(List<Double> inputSignals) {
        for (int i = 0; i < inputSignals.size(); i++) {
            //each input neuron accepts only one input signal
            List<Double> signal = new ArrayList<>(Collections.singletonList(inputSignals.get(i)));
            //getting neuron
            Neuron neuron = layers.get(0).getNeuron(i);
            //sending a signal to neuron
            neuron.feedForward(signal);
        }
    }

    private void feedForwardAllLayersAfterInput() {
        for (int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            //getting list of outputs signals with previous layer
            List<Double> previousLayerSignals = layers.get(i - 1).getOutputSignals();
            for (Neuron neuron : layer.getNeurons()) {
                //sending input signals to neuron
                neuron.feedForward(previousLayerSignals);
            }
        }
    }

    //method for the correction of weights
    public void learnBackPropagation(DataSet dataSet, int numberOfSteps) throws MultiLayerPerceptronException {
        if (dataSet.getInputCount() != topology.getInputCount()) {
            throw new MultiLayerPerceptronException("The number of input signals in dataset not equals " +
                    "to the number of inputs of the trained neural network!");
        }
        lastDataSet = dataSet;
        if (scalingData) {
            //scaling dataset
            lastDataSet.scale();
        }
        for (int i = 0; i < numberOfSteps; i++) {
            //going trough dataset
            for (int j = 0; j < dataSet.getSize(); j++) {
                //getting lists of input signal and expected results from dataset
                List<Double> inputs = lastDataSet.getInputSignals(j);
                List<Double> expectedResults = lastDataSet.getExpectedResult(j);
                //first - calculate result, using special method without scaling inputs,
                //because dataset is already scaled
                //second - calculate errors
                //third - correct weights
                this.predictLearning(inputs);
                this.calculateErrors(expectedResults);
                this.updateWeights(topology.getLearningRate());
            }
        }
    }

    private void calculateErrors(List<Double> expectedResults) {
        //errors calculate from output layer to input
        //because errors in this layer depend on the errors of the next layer
        for (int i = layers.size() - 1; i >= 0; i--) {
            //get layer
            Layer currentLayer = layers.get(i);
            for (int j = 0; j < currentLayer.getNeurons().size(); j++) {
                //for output layer pass a expected result
                if (currentLayer.getLayerType() == NeuronType.Output) {
                    currentLayer.getNeuron(j).calculateError(expectedResults.get(j));
                } else {
                    //for another layers - pass a next layer for calculating the error
                    currentLayer.getNeuron(j).calculateError(layers.get(i + 1), j);
                }
            }
        }
    }

    private void updateWeights(double learningRate) {
        //weights update from output layer to input
        for (int i = layers.size() - 1; i >= 0; i--) {
            //getting layer
            Layer currentLayer = layers.get(i);
            for (int j = 0; j < currentLayer.getNeurons().size(); j++) {
                //update weights of all neurons except input
                if (currentLayer.getLayerType() != NeuronType.Input) {
                    currentLayer.getNeuron(j).learnBackPropagation(learningRate);
                }
            }
        }
    }

    //this method allowed to save the neural network to the specified file
    public void save(String path) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(path);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(this);
        objectOutputStream.close();
        fileOutputStream.close();
    }

    //this method allowed to load the neural network from the specified file
    public static MultiLayerPerceptron load(String path) throws IOException, ClassNotFoundException {
        FileInputStream fileInputStream = new FileInputStream(path);
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        MultiLayerPerceptron MLP = (MultiLayerPerceptron) objectInputStream.readObject();
        objectInputStream.close();
        fileInputStream.close();
        return MLP;
    }
}

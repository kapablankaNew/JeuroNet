package kapablankaNew.JeuroNet.MLP;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode
public class Layer implements Serializable {
    //list of neurons in this layer
    @Getter
    private final List<Neuron> neurons;

    @Getter
    private final NeuronType layerType;

    @Getter
    private final ActivationFunction activationFunction;

    public Layer(List<Neuron> neurons, NeuronType type, ActivationFunction activationFunction) {
        this.neurons = neurons;
        this.activationFunction = activationFunction;
        layerType = type;
    }

    public Layer(List<Neuron> neurons, NeuronType type) {
        this(neurons, type, ActivationFunction.SIGMOID);
    }

    public Layer(List<Neuron> neurons, ActivationFunction activationFunction) {
        this(neurons, NeuronType.Normal, activationFunction);
    }

    public Layer(List<Neuron> neurons) {
        this(neurons, NeuronType.Normal, ActivationFunction.SIGMOID);
    }

    public Neuron getNeuron(int index) {
        return neurons.get(index);
    }

    public int getCount() {
        return neurons.size();
    }

    //method for combining the output signals of all neurons in this layer into a single list
    public List<Double> getOutputSignals() {
        List<Double> signals = new ArrayList<>();
        for (Neuron neuron : neurons) {
            signals.add(neuron.getOutput());
        }
        return signals;
    }
}

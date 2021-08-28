package kapablankaNew.JeuroNet.MLP;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.TopologyException;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@EqualsAndHashCode
public class Topology implements Serializable {
    //number of inputs in neural network (in first layer)
    @Getter
    private final int inputCount;
    //number of outputs of neural network (ih last layer)
    @Getter
    private final int outputCount;
    //number of neurons in several hidden layer
    @Getter
    private final List<LayerInfo> hiddenLayersInfos;
    @Getter
    private final LayerInfo outputLayerInfo;
    @Getter
    private final double learningRate;

    public Topology(int inputCount, LayerInfo outputLayerInfo, @NonNull List<LayerInfo> layers, double learningRate) throws TopologyException {
        if (inputCount <= 0) {
            throw new TopologyException("Number of inputs must be greater than 0!");
        }
        if (outputLayerInfo.getNumberOfNeurons() <= 0) {
            throw new TopologyException("Number of outputs must be greater than 0!");
        }
        if (learningRate <= 0) {
            throw new TopologyException("Learning rate must be greater than 0!");
        }

        this.inputCount = inputCount;
        this.outputLayerInfo = outputLayerInfo;
        this.outputCount = outputLayerInfo.getNumberOfNeurons();
        this.learningRate = learningRate;

        hiddenLayersInfos = new ArrayList<>();
        hiddenLayersInfos.addAll(layers);
    }

    @Deprecated
    public Topology(int inputCount, int outputCount, @NonNull int[] layers, double learningRate) throws TopologyException {
        if (inputCount <= 0) {
            throw new TopologyException("Number of inputs must be greater than 0!");
        }
        if (outputCount <= 0) {
            throw new TopologyException("Number of outputs must be greater than 0!");
        }
        if (learningRate <= 0) {
            throw new TopologyException("Learning rate must be greater than 0!");
        }

        this.inputCount = inputCount;
        this.outputLayerInfo = new LayerInfo(outputCount, ActivationFunction.RELU);
        this.outputCount = outputLayerInfo.getNumberOfNeurons();
        this.learningRate = learningRate;

        hiddenLayersInfos = new ArrayList<>();
        Arrays.stream(layers).forEach(i -> hiddenLayersInfos.add(new LayerInfo(i, ActivationFunction.SIGMOID)));

        List<LayerInfo> layerInfoList = new ArrayList<>();

        for (int layer : layers) {
            layerInfoList.add(new LayerInfo(layer, ActivationFunction.SIGMOID));
        }
    }

    @Deprecated
    public Topology(int inputCount, int outputCount, int layer, double learningRate) throws TopologyException {
        this(inputCount, outputCount, new int[]{layer}, learningRate);
    }

    @Deprecated
    public Topology(int inputCount, int outputCount, double learningRate) throws TopologyException {
        this(inputCount, outputCount, new int[]{outputCount}, learningRate);
    }

    public int getCountOfNeuronsInLayer(int index) {
        return hiddenLayersInfos.get(index).getNumberOfNeurons();
    }
}

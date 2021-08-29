package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.LossFunction;
import kapablankaNew.JeuroNet.TopologyException;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;

@EqualsAndHashCode
public class RNNTopology implements Serializable {
    @Getter
    private final int inputSize;

    @Getter
    private final int outputCount;

    @Getter
    private final int outputSize;

    @Getter
    private final int hiddenCount;

    @Getter
    private final double learningRate;

    @Getter
    private final ActivationFunction activationFunction;

    @Getter
    private final LossFunction lossFunction;

    public RNNTopology(int inputSize, int outputCount, int outputSize, int hiddenCount,
                       double learningRate, ActivationFunction activationFunction,
                       LossFunction lossFunction) throws TopologyException {
        if (outputCount <= 0) {
            throw new TopologyException("Number of outputs must be greater than 0!");
        }
        if (inputSize <= 0) {
            throw new TopologyException("Size of the input data must be greater than 0!");
        }
        if (outputSize <= 0) {
            throw new TopologyException("Size of the output data must be greater than 0!");
        }
        if (hiddenCount <= 0) {
            throw new TopologyException("Hidden count must be greater than 0!");
        }
        if (learningRate <= 0.0) {
            throw new TopologyException("Learning rate must be greater than 0!");
        }
        this.inputSize = inputSize;
        this.outputCount = outputCount;
        this.outputSize = outputSize;
        this.hiddenCount = hiddenCount;
        this.learningRate = learningRate;
        this.lossFunction = lossFunction;
        this.activationFunction = activationFunction;
    }
}

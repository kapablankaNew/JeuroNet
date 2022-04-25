package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.TopologyException;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;

@EqualsAndHashCode
public abstract class AbstractRecurrentLayerTopology implements RecurrentLayerTopology, Serializable {
    @Getter
    private final int inputSize;

    @Getter
    private final int outputCount;

    @Getter
    private final int outputSize;

    @Getter
    private final double learningRate;

    @Getter
    private final RecurrentLayerType recurrentLayerType;

    protected AbstractRecurrentLayerTopology(int inputSize, int outputCount, int outputSize, double learningRate,
                                           RecurrentLayerType recurrentLayerType) throws TopologyException {
        if (outputCount <= 0) {
            throw new TopologyException("Number of outputs must be greater than 0!");
        }
        if (inputSize <= 0) {
            throw new TopologyException("Size of the input data must be greater than 0!");
        }
        if (outputSize <= 0) {
            throw new TopologyException("Size of the output data must be greater than 0!");
        }
        if (learningRate <= 0.0) {
            throw new TopologyException("Learning rate must be greater than 0!");
        }
        this.inputSize = inputSize;
        this.outputCount = outputCount;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.recurrentLayerType = recurrentLayerType;
    }
}

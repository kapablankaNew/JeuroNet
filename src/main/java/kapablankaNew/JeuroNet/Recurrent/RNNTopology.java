package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.TopologyException;
import lombok.Getter;

public class RNNTopology {
    @Getter
    private final int inputCount;

    @Getter
    private final int inputSize;

    @Getter
    private final int outputCount;

    @Getter
    private final int outputSize;

    @Getter
    private final int hiddenCount;

    public RNNTopology(int inputCount, int inputSize, int outputCount, int outputSize, int hiddenCount) throws TopologyException {
        if (inputCount <= 0) {
            throw new TopologyException("Number of inputs must be greater than 0!");
        }
        if (outputCount <= 0) {
            throw new TopologyException("Number of outputs must be greater than 0!");
        }
        if (inputSize <= 0) {
            throw new TopologyException("Size of the input data must be greater than 0!");
        }
        if (outputSize <= 0) {
            throw new TopologyException("Size of the output data must be greater than 0!");
        }
        if (hiddenCount < inputCount || hiddenCount < outputCount) {
            throw new TopologyException("Size of the hidden data must be greater than input and output data!");
        }
        this.inputCount = inputCount;
        this.inputSize = inputSize;
        this.outputCount = outputCount;
        this.outputSize = outputSize;
        this.hiddenCount = hiddenCount;
    }
}

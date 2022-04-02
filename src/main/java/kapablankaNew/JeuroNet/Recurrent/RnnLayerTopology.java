package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.TopologyException;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;

@EqualsAndHashCode(callSuper = true)
public class RnnLayerTopology extends AbstractRecurrentLayerTopology implements Serializable {
    @Getter
    private final int hiddenSize;

    @Getter
    private final ActivationFunction activationFunction;

    @Builder
    private RnnLayerTopology(int inputSize, int outputCount, int outputSize, int hiddenSize,
                             double learningRate, ActivationFunction activationFunction,
                             RecurrentLayerType recurrentLayerType) throws TopologyException {
        super(inputSize, outputCount, outputSize, learningRate, recurrentLayerType);
        if (hiddenSize <= 0) {
            throw new TopologyException("Hidden count must be greater than 0!");
        }
        this.hiddenSize = hiddenSize;
        this.activationFunction = activationFunction;
    }

    @Override
    public RecurrentLayer createRecurrentLayer() throws VectorMatrixException {
        return new RnnLayer(this);
    }
}

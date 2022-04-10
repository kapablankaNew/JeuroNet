package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.TopologyException;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;

@EqualsAndHashCode(callSuper = true)
public class GruLayerTopology extends AbstractRecurrentLayerTopology implements Serializable {
    @Getter
    private final int hiddenSize;

    @Builder
    private GruLayerTopology(int inputSize, int outputCount, int outputSize, int hiddenSize,
                             double learningRate, RecurrentLayerType recurrentLayerType) throws TopologyException {
        super(inputSize, outputCount, outputSize, learningRate, recurrentLayerType);
        this.hiddenSize = hiddenSize;
    }

    @Override
    public RecurrentLayer createRecurrentLayer() throws VectorMatrixException {
        return new GruLayer(this);
    }
}

package kapablankaNew.JeuroNet.Recurrent.Interfaces;

import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Recurrent.RecurrentLayerType;

public interface RecurrentLayerTopology {
    RecurrentLayer createRecurrentLayer() throws VectorMatrixException;

    int getInputSize();

    int getOutputCount();

    int getOutputSize();

    double getLearningRate();

    RecurrentLayerType getRecurrentLayerType();
}

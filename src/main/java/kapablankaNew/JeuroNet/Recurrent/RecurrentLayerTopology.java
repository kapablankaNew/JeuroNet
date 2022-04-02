package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;

public interface RecurrentLayerTopology {
    RecurrentLayer createRecurrentLayer() throws VectorMatrixException;
}

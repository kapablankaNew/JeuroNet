package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.Matrix;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;

public class GruLayer extends AbstractRecurrentLayer implements Serializable {
    protected GruLayer(@NonNull GruLayerTopology topology) throws VectorMatrixException {
        super(topology);
        Wxz = createWeightsMatrix(topology.getOutputSize(), topology.getInputSize());
        Whz = createWeightsMatrix(topology.getOutputSize(), topology.getOutputSize());
        Wxr = createWeightsMatrix(topology.getOutputSize(), topology.getInputSize());
        Whr = createWeightsMatrix(topology.getOutputSize(), topology.getOutputSize());
        Wxo = createWeightsMatrix(topology.getOutputSize(), topology.getInputSize());
        Who = createWeightsMatrix(topology.getOutputSize(), topology.getOutputSize());
        bz = new Vector(topology.getOutputSize(), VectorType.COLUMN);
        br = new Vector(topology.getOutputSize(), VectorType.COLUMN);
    }

    @Override
    public List<Vector> predict(List<Vector> inputSignals) throws VectorMatrixException {
        return null;
    }

    @Override
    public List<Vector> learn(List<Vector> inputSignals, List<Vector> errorsGradients) throws VectorMatrixException {
        return null;
    }

    private Matrix Wxz, Whz, Wxr, Whr, Wxo, Who;

    private Vector bz, br;

    //Next fields storage data about last feed forward step.
    //These data use in the learning process
    private List<Vector> lastValuesH, lastValuesZ, lastValuesR, lastValuesO;

    private List<Vector> lastValuesZZ, lastValuesRR, lastValuesOO;
}

package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;

import java.util.List;

public interface RecurrentLayer {
    List<Vector> predict (List<Vector> inputSignals) throws VectorMatrixException;

    List<Vector> learn(List<Vector> inputSignals, List<Vector> errorsGradients) throws VectorMatrixException;
}

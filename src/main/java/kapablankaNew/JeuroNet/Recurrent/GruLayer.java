package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.Matrix;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
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
        clearCacheValues();
        lastInputs = new ArrayList<>(inputSignals);
        lastOutputs = new ArrayList<>();
        updateInputs();
        Vector h = new Vector(topology.getOutputSize(), VectorType.COLUMN);
        Vector z, r, o, zz, rr, oo, y;
        // Additional values
        Vector first, second;
        for (Vector inputSignal : lastInputs) {
            // Wxz * xi
            first = Wxz.mul(inputSignal);
            // Whz * h(i-1)
            second = Whz.mul(h);
            // Wxz * xi + Whz * h(i-1) + bz
            zz = first.add(second).add(bz);
            // z = Sigma(Wxz * xi + Whz * h(i-1) + bz)
            z = sigma.function(zz);

            // Wxr * xi
            first = Wxr.mul(inputSignal);
            // Whr * h(i-1)
            second = Whr.mul(h);
            // Wxr * xi + Whr * h(i-1) + br
            rr = first.add(second).add(br);
            // r = Sigma(Wxr * xi + Whr * h(i-1) + br)
            r = sigma.function(rr);

            // Wxo * xi
            first = Wxo.mul(inputSignal);
            // Who * (r ◎ h(i-1))
            second = Who.mul(r.mulElemByElem(h));
            // Wxo * xi + Who * (r ◎ h(i-1))
            oo = first.add(second);
            // o = tanh(Wxo * xi + Who * (r ◎ h(i-1)))
            o = tanh.function(oo);

            Vector one = Vector.getVectorWithElementsOfOne(topology.getOutputSize(), VectorType.COLUMN);

            // (1 - z) ◎ h(i-1)
            first = (one.sub(z)).mulElemByElem(h);
            // z ◎ o
            second = z.mulElemByElem(o);
            // hi = (1 - z) ◎ h(i-1) + z ◎ o
            h = first.add(second);

            y = new Vector(h);

            lastOutputs.add(y);
            lastValuesZZ.add(zz);
            lastValuesZ.add(z);
            lastValuesRR.add(rr);
            lastValuesR.add(r);
            lastValuesOO.add(oo);
            lastValuesO.add(o);
            lastValuesH.add(h);
        }
        updateOutputs();
        return new ArrayList<>(lastOutputs);
    }

    @Override
    public List<Vector> learn(List<Vector> inputSignals, List<Vector> errorsGradients) throws VectorMatrixException {
        return null;
    }

    private void clearCacheValues() {
        lastValuesH = new ArrayList<>();
        lastValuesO = new ArrayList<>();
        lastValuesZ = new ArrayList<>();
        lastValuesR = new ArrayList<>();

        lastValuesOO = new ArrayList<>();
        lastValuesZZ = new ArrayList<>();
        lastValuesRR = new ArrayList<>();
    }

    private final ActivationFunction sigma = ActivationFunction.SIGMOID;

    private final ActivationFunction tanh = ActivationFunction.TANH;

    private Matrix Wxz, Whz, Wxr, Whr, Wxo, Who;

    private Vector bz, br;

    //Next fields storage data about last feed forward step.
    //These data use in the learning process
    private List<Vector> lastValuesH, lastValuesZ, lastValuesR, lastValuesO;

    private List<Vector> lastValuesZZ, lastValuesRR, lastValuesOO;
}

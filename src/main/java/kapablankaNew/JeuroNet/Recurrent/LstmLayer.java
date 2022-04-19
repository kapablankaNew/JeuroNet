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

public class LstmLayer extends AbstractRecurrentLayer implements Serializable {
    protected LstmLayer(@NonNull LstmLayerTopology topology) throws VectorMatrixException {
        super(topology);
        hiddenSize = topology.getHiddenSize();
        Wxf = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        Whf = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Wxi = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        Whi = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Wxo = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        Who = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Wxz = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        Whz = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Why = createWeightsMatrix(topology.getOutputSize(), topology.getHiddenSize());
        bf = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        bi = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        bo = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        bz = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        by = new Vector(topology.getOutputSize(), VectorType.COLUMN);
    }

    @Override
    public List<Vector> predict(List<Vector> inputSignals) throws VectorMatrixException {
        clearCacheValues();
        lastInputs = new ArrayList<>(inputSignals);
        lastOutputs = new ArrayList<>();
        updateInputs();
        Vector h = new Vector(hiddenSize, VectorType.COLUMN);
        Vector c = new Vector(hiddenSize, VectorType.COLUMN);
        Vector f, i, o, z, ff, ii, oo, zz, cc, y;
        // Additional values
        Vector first, second;
        for (Vector inputSignal : lastInputs) {
            // Wxf * xi
            first = Wxf.mul(inputSignal);
            // Whf * h(i-1)
            second = Whf.mul(h);
            // Wxf * xi + Whf * h(i-1) + bf
            ff = first.add(second).add(bf);
            // f = Sigma(Wxf * xi + Whf * h(i-1) + bf)
            f = sigma.function(ff);

            // Wxi * xi
            first = Wxi.mul(inputSignal);
            // Whi * h(i-1)
            second = Whi.mul(h);
            // Wxi * xi + Whi * h(i-1) + bi
            ii = first.add(second).add(bi);
            // i = Sigma(Wxi * xi + Whi * h(i-1) + bi)
            i = sigma.function(ii);

            // Wxo * xi
            first = Wxo.mul(inputSignal);
            // Who * h(i-1)
            second = Who.mul(h);
            // Wxo * xi + Who * h(i-1) + bo
            oo = first.add(second).add(bo);
            // o = Sigma(Wxo * xi + Who * h(i-1) + bo)
            o = sigma.function(oo);

            // Wxz * xi
            first = Wxz.mul(inputSignal);
            // Whz * h(i-1)
            second = Whz.mul(h);
            // Wxz * xi + Whz * h(i-1) + bz
            zz = first.add(second).add(bz);
            // o = Sigma(Wxo * xi + Who * h(i-1) + bo)
            z = tanh.function(zz);

            // f ◎ c(i-1)
            first = f.mulElemByElem(c);
            // i ◎ z
            second = i.mulElemByElem(z);
            // ci = f ◎ c(i-1) + i ◎ z
            c = first.add(second);
            // cc = tanh(f ◎ c(i-1) + i ◎ z)
            cc = tanh.function(c);

            // hi = o ◎ cc
            h = o.mulElemByElem(cc);

            y = (Why.mul(h)).add(by);

            lastOutputs.add(y);

            lastValuesC.add(c);
            lastValuesH.add(h);
            lastValuesF.add(f);
            lastValuesI.add(i);
            lastValuesO.add(o);
            lastValuesZ.add(z);

            lastValuesFF.add(ff);
            lastValuesII.add(ii);
            lastValuesOO.add(oo);
            lastValuesZZ.add(zz);
            lastValuesCC.add(cc);
        }
        updateOutputs();
        return new ArrayList<>(lastOutputs);
    }

    @Override
    public List<Vector> learn(List<Vector> inputSignals, List<Vector> errorsGradients) throws VectorMatrixException {
        return null;
    }

    private void clearCacheValues() {
        lastValuesC = new ArrayList<>();
        lastValuesH = new ArrayList<>();
        lastValuesF = new ArrayList<>();
        lastValuesI = new ArrayList<>();
        lastValuesO = new ArrayList<>();
        lastValuesZ = new ArrayList<>();

        lastValuesFF = new ArrayList<>();
        lastValuesII = new ArrayList<>();
        lastValuesOO = new ArrayList<>();
        lastValuesZZ = new ArrayList<>();
        lastValuesCC = new ArrayList<>();
    }

    private final ActivationFunction sigma = ActivationFunction.SIGMOID;

    private final ActivationFunction tanh = ActivationFunction.TANH;

    private Matrix Wxf, Whf, Wxi, Whi, Wxo, Who, Wxz, Whz, Why;

    private Vector bf, bi, bo, bz, by;

    //Next fields storage data about last feed forward step.
    //These data use in the learning process
    private List<Vector> lastValuesC, lastValuesH, lastValuesF, lastValuesI, lastValuesO, lastValuesZ;

    private List<Vector> lastValuesFF, lastValuesII, lastValuesOO, lastValuesZZ, lastValuesCC;

    private final int hiddenSize;
}

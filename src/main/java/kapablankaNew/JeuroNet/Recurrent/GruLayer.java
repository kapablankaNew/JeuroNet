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
        Wxz = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        Whz = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Wxr = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Whr = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Wxo = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        Who = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Why = createWeightsMatrix(topology.getOutputSize(), topology.getHiddenSize());
        bz = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        br = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        by = new Vector(topology.getOutputSize(), VectorType.COLUMN);
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

            y = (Why.mul(h)).add(by);

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
        predict(inputSignals);
        double learningRate = topology.getLearningRate();
        List<Vector> resultErrorsGradients = new ArrayList<>();
        for (int i = 0; i < inputSignals.size(); i++)
        {
            resultErrorsGradients.add(new Vector(topology.getInputSize()));
        }
        //create matrices for gradients
        Matrix d_Why = new Matrix(Why.getRows(), Why.getColumns());
        Matrix d_Who = new Matrix(Who.getRows(), Who.getColumns());
        Matrix d_Wxo = new Matrix(Wxo.getRows(), Wxo.getColumns());
        Matrix d_Whr = new Matrix(Whr.getRows(), Whr.getColumns());
        Matrix d_Wxr = new Matrix(Wxr.getRows(), Wxr.getColumns());
        Matrix d_Whz = new Matrix(Whz.getRows(), Whz.getColumns());
        Matrix d_Wxz = new Matrix(Wxz.getRows(), Wxz.getColumns());
        Vector d_bz = new Vector(bz.size(), bz.getType());
        Vector d_br = new Vector(br.size(), br.getType());
        Vector d_by = new Vector(by.size(), by.getType());

        for (int i = 0; i < errorsGradients.size(); i++) {
            //first - get loss function gradient
            Vector d_y = new Vector(errorsGradients.get(errorsGradients.size() - 1 - i));

            //calculate values dE/dby and dE/dWhy
            d_Why = d_Why.add(d_y.mul(lastValuesH.get(lastValuesH.size() - 1 - i).T()));
            d_by = d_by.add(d_y);

            ////it is special value: dE/dzi
            //Vector temp = Why.T().mul(d_y).mulElemByElem(AF.derivative(lastValuesZ.get(lastValuesZ.size() - 1)));
//
            //for (int k = inputSignals.size() - 1 - i; k >= 0; k--) {
            //    Vector h_k = lastValuesH.get(k);
            //    Vector z_k = lastValuesZ.get(k);
//
            //    //update gradient values
            //    d_Whh = d_Whh.add(temp.mul(h_k.T()));
            //    d_Wxh = d_Wxh.add(temp.mul(lastInputs.get(k).T()));
            //    d_bh = d_bh.add(temp);
//
            //    //update dE/dxi
            //    if(k < inputSignals.size()) {
            //        Vector d_x = (temp.T().mul(Wxh)).T();
            //        resultErrorsGradients.set(k, resultErrorsGradients.get(k).add(d_x));
            //    }
//
            //    //dE/dzi = Whh.T * dE/dz(i+1) * dh(i+1)/dzi
            //    temp = Whh.T().mul(temp.mulElemByElem(AF.derivative(z_k)));
            //}
        }
        //limit the values in gradients to avoid the problem of vanishing gradients
        d_Why = d_Why.limit(-1.0, 1.0);
        d_by = d_by.limit(-1.0, 1.0);

        d_Who = d_Who.limit(-1.0, 1.0);
        d_Wxo = d_Wxo.limit(-1.0, 1.0);

        d_Whr = d_Whr.limit(-1.0, 1.0);
        d_Wxr = d_Wxr.limit(-1.0, 1.0);
        d_br = d_br.limit(-1.0, 1.0);

        d_Whz = d_Whz.limit(-1.0, 1.0);
        d_Wxz = d_Wxz.limit(-1.0, 1.0);
        d_bz = d_bz.limit(-1.0, 1.0);

        for (int i = 0; i < resultErrorsGradients.size(); i++) {
            resultErrorsGradients.set(i, resultErrorsGradients.get(i).limit(-1.0, 1.0));
        }
        //update parameters of RNN
        Why = Why.sub(d_Why.mul(learningRate));
        by = by.sub(d_by.mul(learningRate));

        Who = Who.sub(d_Who.mul(learningRate));
        Wxo = Wxo.sub(d_Wxo.mul(learningRate));

        Whr = Whr.sub(d_Whr.mul(learningRate));
        Wxr = Wxr.sub(d_Wxr.mul(learningRate));
        br = br.sub(d_br.mul(learningRate));

        Whz = Whz.sub(d_Whz.mul(learningRate));
        Wxz = Wxz.sub(d_Wxz.mul(learningRate));
        bz = bz.sub(d_bz.mul(learningRate));

        return resultErrorsGradients;
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

    private Matrix Wxz, Whz, Wxr, Whr, Wxo, Who, Why;

    private Vector bz, br, by;

    //Next fields storage data about last feed forward step.
    //These data use in the learning process
    private List<Vector> lastValuesH, lastValuesZ, lastValuesR, lastValuesO;

    private List<Vector> lastValuesZZ, lastValuesRR, lastValuesOO;
}

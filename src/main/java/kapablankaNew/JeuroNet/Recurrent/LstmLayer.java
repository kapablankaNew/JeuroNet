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
        lastValuesC.add(c);
        lastValuesH.add(h);
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
        predict(inputSignals);
        double learningRate = topology.getLearningRate();
        List<Vector> resultErrorsGradients = new ArrayList<>();
        for (int i = 0; i < inputSignals.size(); i++)
        {
            resultErrorsGradients.add(new Vector(topology.getInputSize()));
        }
        //create matrices for gradients
        Matrix d_Wxf = new Matrix(Wxf.getRows(), Wxf.getColumns());
        Matrix d_Whf = new Matrix(Whf.getRows(), Whf.getColumns());
        Matrix d_Wxi = new Matrix(Wxi.getRows(), Wxi.getColumns());
        Matrix d_Whi = new Matrix(Whi.getRows(), Whi.getColumns());
        Matrix d_Wxo = new Matrix(Wxo.getRows(), Wxo.getColumns());
        Matrix d_Who = new Matrix(Who.getRows(), Who.getColumns());
        Matrix d_Wxz = new Matrix(Wxz.getRows(), Wxz.getColumns());
        Matrix d_Whz = new Matrix(Whz.getRows(), Whz.getColumns());
        Matrix d_Why = new Matrix(Why.getRows(), Why.getColumns());

        Vector d_bf = new Vector(bf.size(), bf.getType());
        Vector d_bi = new Vector(bi.size(), bi.getType());
        Vector d_bo = new Vector(bo.size(), bo.getType());
        Vector d_bz = new Vector(bz.size(), bz.getType());
        Vector d_by = new Vector(by.size(), by.getType());

        for (int j = 0; j < errorsGradients.size(); j++) {
            //first - get loss function gradient
            Vector dL_dy = new Vector(errorsGradients.get(errorsGradients.size() - 1 - j));

            //calculate values dE/dby and dE/dWhy
            d_Why = d_Why.add(dL_dy.mul(lastValuesH.get(lastValuesH.size() - 1 - j).T()));
            d_by = d_by.add(dL_dy);

            //it is special value: dE/dzi
            Vector dL_dh = Why.T().mul(dL_dy);
            Vector dL_dc = new Vector(hiddenSize, VectorType.COLUMN);

            for (int k = inputSignals.size() - 1 - j; k >= 0; k--) {
                Vector x = inputSignals.get(k);
                Vector h_prev = lastValuesH.get(k);
                Vector cc = lastValuesCC.get(k);
                Vector o = lastValuesO.get(k);
                Vector oo = lastValuesOO.get(k);
                Vector c = lastValuesC.get(k + 1);
                Vector c_prev = lastValuesC.get(k);
                Vector z = lastValuesZ.get(k);
                Vector zz = lastValuesZZ.get(k);
                Vector f = lastValuesF.get(k);
                Vector ff = lastValuesFF.get(k);
                Vector i = lastValuesI.get(k);
                Vector ii = lastValuesII.get(k);

                Vector dL_do = cc.mulElemByElem(dL_dh);
                Vector dL_doo = dL_do.mulElemByElem(sigma.derivative(oo));

                Vector dL_dcc = o.mulElemByElem(dL_dh);
                dL_dc = dL_dcc.mulElemByElem(tanh.derivative(c)).add(dL_dc);

                Vector dL_dz = i.mulElemByElem(dL_dc);
                Vector dL_dzz = dL_dz.mulElemByElem(tanh.derivative(zz));

                Vector dL_df = c_prev.mulElemByElem(dL_dc);
                Vector dL_dff = dL_df.mulElemByElem(sigma.derivative(ff));

                Vector dL_di = z.mulElemByElem(dL_dc);
                Vector dL_dii = dL_di.mulElemByElem(sigma.derivative(ii));

                dL_dc = dL_dc.mulElemByElem(f);

                Matrix dL_dWxf = dL_dff.mul(x.T());
                Matrix dL_dWhf = dL_dff.mul(h_prev.T());
                Vector dL_dbf = new Vector(dL_dff);

                Matrix dL_dWxi = dL_dii.mul(x.T());
                Matrix dL_dWhi = dL_dii.mul(h_prev.T());
                Vector dL_dbi = new Vector(dL_dii);

                Matrix dL_dWxo = dL_doo.mul(x.T());
                Matrix dL_dWho = dL_doo.mul(h_prev.T());
                Vector dL_dbo = new Vector(dL_doo);

                Matrix dL_dWxz = dL_dzz.mul(x.T());
                Matrix dL_dWhz = dL_dzz.mul(h_prev.T());
                Vector dL_dbz = new Vector(dL_dzz);

                d_Wxf = d_Wxf.add(dL_dWxf);
                d_Whf = d_Whf.add(dL_dWhf);
                d_Wxi = d_Wxi.add(dL_dWxi);
                d_Whi = d_Whi.add(dL_dWhi);
                d_Wxo = d_Wxo.add(dL_dWxo);
                d_Who = d_Who.add(dL_dWho);
                d_Wxz = d_Wxz.add(dL_dWxz);
                d_Whz = d_Whz.add(dL_dWhz);

                d_bf = d_bf.add(dL_dbf);
                d_bi = d_bi.add(dL_dbi);
                d_bo = d_bo.add(dL_dbo);
                d_bz = d_bz.add(dL_dbz);

                // Update dL/dx
                if (k < inputSignals.size()) {
                    Vector first = Wxo.T().mul(dL_doo);
                    Vector second = Wxz.T().mul(dL_dzz);
                    Vector third = Wxf.T().mul(dL_dff);
                    Vector fourth = Wxi.T().mul(dL_dii);
                    Vector dL_dx = first.add(second).add(third).add(fourth);
                    resultErrorsGradients.set(k, resultErrorsGradients.get(k).add(dL_dx));
                }

                // Update dL/dh
                Vector first = Whi.T().mul(dL_dii);
                Vector second = Whf.T().mul(dL_dff);
                Vector third = Whz.T().mul(dL_dzz);
                Vector fourth = Who.T().mul(dL_doo);
                dL_dh = first.add(second).add(third).add(fourth);
            }
        }
        //limit the values in gradients to avoid the problem of vanishing gradients
        d_Why = d_Why.limit(-1.0, 1.0);
        d_by = d_by.limit(-1.0, 1.0);

        d_Whf = d_Whf.limit(-1.0, 1.0);
        d_Wxf = d_Wxf.limit(-1.0, 1.0);
        d_bf = d_bf.limit(-1.0, 1.0);

        d_Whi = d_Whi.limit(-1.0, 1.0);
        d_Wxi = d_Wxi.limit(-1.0, 1.0);
        d_bi = d_bi.limit(-1.0, 1.0);

        d_Who = d_Who.limit(-1.0, 1.0);
        d_Wxo = d_Wxo.limit(-1.0, 1.0);
        d_bo = d_bo.limit(-1.0, 1.0);

        d_Whz = d_Whz.limit(-1.0, 1.0);
        d_Wxz = d_Wxz.limit(-1.0, 1.0);
        d_bz = d_bz.limit(-1.0, 1.0);

        for (int i = 0; i < resultErrorsGradients.size(); i++) {
            resultErrorsGradients.set(i, resultErrorsGradients.get(i).limit(-1.0, 1.0));
        }
        //update parameters of LSTM
        Why = Why.sub(d_Why.mul(learningRate));
        by = by.sub(d_by.mul(learningRate));

        Whf = Whf.sub(d_Whf.mul(learningRate));
        Wxf = Wxf.sub(d_Wxf.mul(learningRate));
        bf = bf.sub(d_bf.mul(learningRate));

        Whi = Whi.sub(d_Whi.mul(learningRate));
        Wxi = Wxi.sub(d_Wxi.mul(learningRate));
        bi = bi.sub(d_bi.mul(learningRate));

        Who = Who.sub(d_Who.mul(learningRate));
        Wxo = Wxo.sub(d_Wxo.mul(learningRate));
        bo = bo.sub(d_bo.mul(learningRate));

        Whz = Whz.sub(d_Whz.mul(learningRate));
        Wxz = Wxz.sub(d_Wxz.mul(learningRate));
        bz = bz.sub(d_bz.mul(learningRate));

        return resultErrorsGradients;
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

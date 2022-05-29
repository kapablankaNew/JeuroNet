package kapablankaNew.JeuroNet.Recurrent.GRU;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.Matrix;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
import kapablankaNew.JeuroNet.Recurrent.Interfaces.AbstractRecurrentLayer;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode(callSuper = true)
public class GruLayer extends AbstractRecurrentLayer implements Serializable {
    protected GruLayer(@NonNull GruLayerTopology topology) throws VectorMatrixException {
        super(topology);
        hiddenSize = topology.getHiddenSize();
        parameters = new GruParameters();
        parameters.Wxz = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        parameters.Whz = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        parameters.Wxr = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        parameters.Whr = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        parameters.Wxo = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        parameters.Who = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        parameters.Why = createWeightsMatrix(topology.getOutputSize(), topology.getHiddenSize());
        parameters.bz = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        parameters.br = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        parameters.bo = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        parameters.by = new Vector(topology.getOutputSize(), VectorType.COLUMN);
    }

    @Override
    public synchronized List<Vector> predict(List<Vector> inputSignals) throws VectorMatrixException {
        Matrix Wxz = parameters.Wxz;
        Matrix Whz = parameters.Whz;
        Vector bz = parameters.bz;
        Matrix Wxr = parameters.Wxr;
        Matrix Whr = parameters.Whr;
        Vector br = parameters.br;
        Matrix Wxo = parameters.Wxo;
        Matrix Who = parameters.Who;
        Vector bo = parameters.bo;
        Matrix Why = parameters.Why;
        Vector by = parameters.by;

        lastInputs = new ArrayList<>(inputSignals);
        lastOutputs = new ArrayList<>();
        updateInputs();
        Vector h = new Vector(hiddenSize, VectorType.COLUMN);
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
            // Wxo * xi + Who * (r ◎ h(i-1)) + bo
            oo = first.add(second).add(bo);
            // o = tanh(Wxo * xi + Who * (r ◎ h(i-1)))
            o = tanh.function(oo);

            Vector one = Vector.getVectorWithElementsOfOne(hiddenSize, VectorType.COLUMN);

            // (1 - z) ◎ h(i-1)
            first = (one.sub(z)).mulElemByElem(h);
            // z ◎ o
            second = z.mulElemByElem(o);
            // hi = (1 - z) ◎ h(i-1) + z ◎ o
            h = first.add(second);

            y = (Why.mul(h)).add(by);

            lastOutputs.add(y);
        }
        updateOutputs();
        return new ArrayList<>(lastOutputs);
    }

    private GruCache predictInLearning(List<Vector> inputSignals, GruParameters localParameters) throws VectorMatrixException {
        Matrix Wxz = new Matrix(localParameters.Wxz);
        Matrix Whz = new Matrix(localParameters.Whz);
        Vector bz = new Vector(localParameters.bz);
        Matrix Wxr = new Matrix(localParameters.Wxr);
        Matrix Whr = new Matrix(localParameters.Whr);
        Vector br = new Vector(localParameters.br);
        Matrix Wxo = new Matrix(localParameters.Wxo);
        Matrix Who = new Matrix(localParameters.Who);
        Vector bo = new Vector(localParameters.bo);
        Matrix Why = new Matrix(localParameters.Why);
        Vector by = new Vector(localParameters.by);
        var cache = new GruCache();

        cache.lastInputs = updateInputs(inputSignals);
        List<Vector> outputs = new ArrayList<>();
        Vector h = new Vector(hiddenSize, VectorType.COLUMN);
        cache.lastValuesH.add(h);
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
            // Wxo * xi + Who * (r ◎ h(i-1)) + bo
            oo = first.add(second).add(bo);
            // o = tanh(Wxo * xi + Who * (r ◎ h(i-1)))
            o = tanh.function(oo);

            Vector one = Vector.getVectorWithElementsOfOne(hiddenSize, VectorType.COLUMN);

            // (1 - z) ◎ h(i-1)
            first = (one.sub(z)).mulElemByElem(h);
            // z ◎ o
            second = z.mulElemByElem(o);
            // hi = (1 - z) ◎ h(i-1) + z ◎ o
            h = first.add(second);

            y = (Why.mul(h)).add(by);

            outputs.add(y);
            cache.lastValuesZZ.add(zz);
            cache.lastValuesZ.add(z);
            cache.lastValuesRR.add(rr);
            cache.lastValuesR.add(r);
            cache.lastValuesOO.add(oo);
            cache.lastValuesO.add(o);
            cache.lastValuesH.add(h);
        }
        cache.lastOutputs = updateOutputs(outputs);
        return cache;
    }

    @Override
    public synchronized List<Vector> learn(List<Vector> inputSignals, List<Vector> errorsGradients) throws VectorMatrixException {
        Matrix Wxz = parameters.Wxz;
        Matrix Whz = parameters.Whz;
        Vector bz = parameters.bz;
        Matrix Wxr = parameters.Wxr;
        Matrix Whr = parameters.Whr;
        Vector br = parameters.br;
        Matrix Wxo = parameters.Wxo;
        Matrix Who = parameters.Who;
        Vector bo = parameters.bo;
        Matrix Why = parameters.Why;
        Vector by = parameters.by;
        GruParameters localParameters = GruParameters.builder()
                .Wxz(Wxz)
                .Whz(Whz)
                .bz(bz)
                .Wxr(Wxr)
                .Whr(Whr)
                .br(br)
                .Wxo(Wxo)
                .Who(Who)
                .bo(bo)
                .Why(Why)
                .by(by)
                .build();
        GruCache cache = predictInLearning(inputSignals, localParameters);
        List<Vector> lastValuesH = cache.lastValuesH;
        List<Vector> lastValuesZ = cache.lastValuesZ;
        List<Vector> lastValuesR = cache.lastValuesR;
        List<Vector> lastValuesO = cache.lastValuesO;
        List<Vector> lastValuesZZ = cache.lastValuesZZ;
        List<Vector> lastValuesRR = cache.lastValuesRR;
        List<Vector> lastValuesOO = cache.lastValuesOO;

        double learningRate = topology.getLearningRate();
        List<Vector> resultErrorsGradients = new ArrayList<>();
        for (int i = 0; i < inputSignals.size(); i++) {
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
        Vector d_bo = new Vector(bo.size(), bo.getType());
        Vector d_by = new Vector(by.size(), by.getType());

        for (int i = 0; i < errorsGradients.size(); i++) {
            //first - get loss function gradient
            Vector dL_dy = new Vector(errorsGradients.get(errorsGradients.size() - 1 - i));

            //calculate values dE/dby and dE/dWhy
            d_Why = d_Why.add(dL_dy.mul(lastValuesH.get(lastValuesH.size() - 1 - i).T()));
            d_by = d_by.add(dL_dy);

            //it is special value: dE/dzi
            Vector dL_dh = Why.T().mul(dL_dy);

            for (int k = inputSignals.size() - 1 - i; k >= 0; k--) {
                Vector x = inputSignals.get(k);
                Vector o = lastValuesO.get(k);
                Vector z = lastValuesZ.get(k);
                Vector r = lastValuesR.get(k);
                Vector h_prev = lastValuesH.get(k);
                Vector zz = lastValuesZZ.get(k);
                Vector rr = lastValuesRR.get(k);
                Vector oo = lastValuesOO.get(k);

                Vector dL_dz = o.sub(h_prev).mulElemByElem(dL_dh);
                Vector dL_dzz = dL_dz.mulElemByElem(sigma.derivative(zz));

                Vector dL_do = z.mulElemByElem(dL_dh);
                Vector dL_doo = dL_do.mulElemByElem(tanh.derivative(oo));

                Vector dL_dr = Who.T().mul(dL_doo.mulElemByElem(h_prev));
                Vector dL_drr = sigma.derivative(rr).mulElemByElem(dL_dr);

                Matrix dL_dWxo = dL_doo.mul(x.T());
                Matrix dL_dWho = dL_doo.mul((r.mulElemByElem(h_prev)).T());
                Vector dL_dbo = new Vector(dL_doo);

                Matrix dL_dWxr = dL_drr.mul(x.T());
                Matrix dL_dWhr = dL_drr.mul(h_prev.T());
                Vector dL_dbr = new Vector(dL_drr);

                Matrix dL_dWxz = dL_dzz.mul(x.T());
                Matrix dL_dWhz = dL_dzz.mul(h_prev.T());
                Vector dL_dbz = new Vector(dL_dzz);

                d_Who = d_Who.add(dL_dWho);
                d_Wxo = d_Wxo.add(dL_dWxo);
                d_Whr = d_Whr.add(dL_dWhr);
                d_Wxr = d_Wxr.add(dL_dWxr);
                d_Whz = d_Whz.add(dL_dWhz);
                d_Wxz = d_Wxz.add(dL_dWxz);
                d_bz = d_bz.add(dL_dbz);
                d_br = d_br.add(dL_dbr);
                d_bo = d_bo.add(dL_dbo);

                // Update dL/dx
                if (k < inputSignals.size()) {
                    Vector first = Wxo.T().mul(dL_doo);
                    Vector second = Wxz.T().mul(dL_dzz);
                    Vector third = Wxr.T().mul(dL_drr);
                    Vector dL_dx = first.add(second).add(third);
                    resultErrorsGradients.set(k, resultErrorsGradients.get(k).add(dL_dx));
                }

                // Update dL/dh
                Vector first = Vector.getVectorWithElementsOfOne(h_prev.size(), h_prev.getType());
                first = first.sub(z).mulElemByElem(dL_dh);
                Vector second = Whr.T().mul(dL_drr);
                Vector third = Who.T().mul(dL_doo.mulElemByElem(r));
                Vector fourth = Whz.T().mul(dL_dzz);
                dL_dh = first.add(second).add(third).add(fourth);
            }
        }
        //limit the values in gradients to avoid the problem of vanishing gradients
        d_Why = d_Why.limit(-1.0, 1.0);
        d_by = d_by.limit(-1.0, 1.0);

        d_Who = d_Who.limit(-1.0, 1.0);
        d_Wxo = d_Wxo.limit(-1.0, 1.0);
        d_bo = d_bo.limit(-1.0, 1.0);

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
        parameters.updateWeights(d_Why, d_Who, d_Wxo, d_Whr, d_Wxr, d_Whz, d_Wxz, d_bz, d_br, d_bo, d_by, learningRate);

        return resultErrorsGradients;
    }

    private final ActivationFunction sigma = ActivationFunction.SIGMOID;

    private final ActivationFunction tanh = ActivationFunction.TANH;

    private final GruParameters parameters;

    private final int hiddenSize;

    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    private static class GruParameters {
        private void updateWeights(Matrix d_Why, Matrix d_Who, Matrix d_Wxo, Matrix d_Whr, Matrix d_Wxr, Matrix d_Whz,
                                   Matrix d_Wxz, Vector d_bz, Vector d_br, Vector d_bo, Vector d_by, double learningRate)
                throws VectorMatrixException {
            Why = Why.sub(d_Why.mul(learningRate));
            by = by.sub(d_by.mul(learningRate));
            Who = Who.sub(d_Who.mul(learningRate));
            Wxo = Wxo.sub(d_Wxo.mul(learningRate));
            bo = bo.sub(d_bo.mul(learningRate));
            Whr = Whr.sub(d_Whr.mul(learningRate));
            Wxr = Wxr.sub(d_Wxr.mul(learningRate));
            br = br.sub(d_br.mul(learningRate));
            Whz = Whz.sub(d_Whz.mul(learningRate));
            Wxz = Wxz.sub(d_Wxz.mul(learningRate));
            bz = bz.sub(d_bz.mul(learningRate));
        }

        private Matrix Wxz, Whz, Wxr, Whr, Wxo, Who, Why;

        private Vector bz, br, bo, by;
    }

    private static class GruCache {

        private GruCache() {
            lastValuesH = new ArrayList<>();
            lastValuesO = new ArrayList<>();
            lastValuesZ = new ArrayList<>();
            lastValuesR = new ArrayList<>();

            lastValuesOO = new ArrayList<>();
            lastValuesZZ = new ArrayList<>();
            lastValuesRR = new ArrayList<>();

            lastInputs = new ArrayList<>();
            lastOutputs = new ArrayList<>();
        }

        @Getter
        private final List<Vector> lastValuesH, lastValuesZ, lastValuesR, lastValuesO;

        @Getter
        private final List<Vector> lastValuesZZ, lastValuesRR, lastValuesOO;

        @Getter
        private List<Vector> lastInputs, lastOutputs;
    }
}

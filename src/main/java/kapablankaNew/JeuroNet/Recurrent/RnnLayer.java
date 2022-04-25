package kapablankaNew.JeuroNet.Recurrent;

/*
This class implements simple recurrent neural network of the RNN type
    yo    y1    y2
    ^     ^     ^
    |     |     |
    h0 -> h1 -> h2 ...
    ^     ^     ^
    |     |     |
    x0    x1    x2

Such network have three weight arrays: Wxh for links (xi -> hi), Whh for links (h(i-1) -> hi), Why for links (hi -> yi)
Also such network have two bias arrays: bh for calculation hi and by for calculation yi
In mathematical form, it look like this:
hi = AF(Wxh * xi + Whh * h(i-1) + bh)
yi = Why * hi + by
AF is activation function, in RNN it's usually tanh.
 */

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.Matrix;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
import lombok.EqualsAndHashCode;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode(callSuper = true)
public class RnnLayer extends AbstractRecurrentLayer implements Serializable {
    public RnnLayer (@NonNull RnnLayerTopology topology) throws VectorMatrixException {
        super(topology);
        activationFunction = topology.getActivationFunction();
        hiddenSize = topology.getHiddenSize();
        Whh = createWeightsMatrix(topology.getHiddenSize(), topology.getHiddenSize());
        Wxh = createWeightsMatrix(topology.getHiddenSize(), topology.getInputSize());
        Why = createWeightsMatrix(topology.getOutputSize(), topology.getHiddenSize());
        bh = new Vector(topology.getHiddenSize(), VectorType.COLUMN);
        by = new Vector(topology.getOutputSize(), VectorType.COLUMN);
    }

    @Override
    public List<Vector> predict(List<Vector> inputSignals) throws VectorMatrixException {
        lastInputs = new ArrayList<>(inputSignals);
        lastOutputs = new ArrayList<>();
        updateInputs();
        lastValuesH = new ArrayList<>();
        lastValuesZ = new ArrayList<>();
        Vector h = new Vector(hiddenSize, VectorType.COLUMN);
        Vector z;
        for (Vector inputSignal : lastInputs) {
            //Wxh * xi
            Vector first = Wxh.mul(inputSignal);
            //Whh * h(i-1)
            Vector second = Whh.mul(h);
            //Wxh * xi + Whh * h(i-1) + bh
            z = first.add(second).add(bh);
            //hi = AF(Wxh * xi + Whh * h(i-1) + bh)
            h = activationFunction.function(z);
            //yi = Why * hi + by
            Vector yi = (Why.mul(h)).add(by);
            lastOutputs.add(yi);
            lastValuesH.add(h);
            lastValuesZ.add(z);
        }
        updateOutputs();
        return new ArrayList<>(lastOutputs);
    }

    @Override
    public List<Vector> learn(List<Vector> inputSignals, List<Vector> errorsGradients) throws VectorMatrixException {
        predict(inputSignals);

        ActivationFunction AF = activationFunction;
        double learningRate = topology.getLearningRate();
        List<Vector> resultErrorsGradients = new ArrayList<>();
        for (int i = 0; i < inputSignals.size(); i++)
        {
            resultErrorsGradients.add(new Vector(topology.getInputSize()));
        }

        //create matrices for gradients
        Matrix d_Why = new Matrix(Why.getRows(), Why.getColumns());
        Matrix d_Whh = new Matrix(Whh.getRows(), Whh.getColumns());
        Matrix d_Wxh = new Matrix(Wxh.getRows(), Wxh.getColumns());
        Vector d_bh = new Vector(bh.size(), bh.getType());
        Vector d_by = new Vector(by.size(), by.getType());
        for (int i = 0; i < errorsGradients.size(); i++) {
            //first - get loss function gradient
            Vector d_y = new Vector(errorsGradients.get(errorsGradients.size() - 1 - i));

            //calculate values dE/dby and dE/dWhy
            d_Why = d_Why.add(d_y.mul(lastValuesH.get(lastValuesH.size() - 1 - i).T()));
            d_by = d_by.add(d_y);

            //it is special value: dE/dzi
            Vector temp = Why.T().mul(d_y).mulElemByElem(AF.derivative(lastValuesZ.get(lastValuesZ.size() - 1 - i)));

            for (int k = inputSignals.size() - 1 - i; k >= 0; k--) {
                Vector h_k = lastValuesH.get(k);
                Vector z_k = lastValuesZ.get(k);

                //update gradient values
                d_Whh = d_Whh.add(temp.mul(h_k.T()));
                d_Wxh = d_Wxh.add(temp.mul(lastInputs.get(k).T()));
                d_bh = d_bh.add(temp);

                //update dE/dxi
                if(k < inputSignals.size()) {
                    Vector d_x = (temp.T().mul(Wxh)).T();
                    resultErrorsGradients.set(k, resultErrorsGradients.get(k).add(d_x));
                }

                //dE/dzi = Whh.T * dE/dz(i+1) * dh(i+1)/dzi
                temp = Whh.T().mul(temp.mulElemByElem(AF.derivative(z_k)));
            }
        }
        //limit the values in gradients to avoid the problem of vanishing gradients
        d_Why = d_Why.limit(-1.0, 1.0);
        d_by = d_by.limit(-1.0, 1.0);

        d_Whh = d_Whh.limit(-1.0, 1.0);
        d_Wxh = d_Wxh.limit(-1.0, 1.0);
        d_bh = d_bh.limit(-1.0, 1.0);

        for (int i = 0; i < resultErrorsGradients.size(); i++) {
            resultErrorsGradients.set(i, resultErrorsGradients.get(i).limit(-1.0, 1.0));
        }
        //update parameters of RNN
        Why = Why.sub(d_Why.mul(learningRate));
        by = by.sub(d_by.mul(learningRate));

        Whh = Whh.sub(d_Whh.mul(learningRate));
        Wxh = Wxh.sub(d_Wxh.mul(learningRate));
        bh = bh.sub(d_bh.mul(learningRate));
        return resultErrorsGradients;
    }

    private Matrix Wxh;

    private Matrix Whh;

    private Matrix Why;

    private Vector bh;

    private Vector by;

    //Next fields storage data about last feed forward step.
    //These data use in the learning process
    @EqualsAndHashCode.Exclude
    private List<Vector> lastValuesH;

    @EqualsAndHashCode.Exclude
    private List<Vector> lastValuesZ;

    private final ActivationFunction activationFunction;

    private final int hiddenSize;
}

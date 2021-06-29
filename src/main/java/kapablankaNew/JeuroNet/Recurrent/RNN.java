package kapablankaNew.JeuroNet.Recurrent;

/*
This class implements simple recurrent neural network, in first variant - only with link "many to many"
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
AF is activation function, in recurrent networks it's usually tanh.
 */

import kapablankaNew.JeuroNet.Mathematical.*;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class RNN {
    private final RNNTopology topology;

    private Matrix Wxh;

    private Matrix Whh;

    private Matrix Why;

    private Vector bh;

    private Vector by;

    //Next two fields storage data about last feed forward step.
    //These data use in the learning process
    private List<Vector> lastInputs;

    private List<Vector> lastHiddenValues;

    public RNN (@NonNull RNNTopology topology) throws VectorMatrixException {
        if (topology.getOutputCount() > 1) {
            throw new IllegalArgumentException("Output count in this version must be 1!");
        }
        this.topology = topology;
        Whh = createWeightsMatrix(topology.getHiddenCount(), topology.getHiddenCount());
        Wxh = createWeightsMatrix(topology.getHiddenCount(), topology.getInputSize());
        Why = createWeightsMatrix(topology.getOutputSize(), topology.getHiddenCount());
        bh = new Vector(topology.getHiddenCount(), VectorType.COLUMN);
        by = new Vector(topology.getOutputSize(), VectorType.COLUMN);
    }

    private static Matrix createWeightsMatrix(int rows, int columns) throws VectorMatrixException {
        List<List<Double>> elements = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                row.add(ThreadLocalRandom.current().nextDouble(0.0, 0.01));
            }
            elements.add(row);
        }
        return new Matrix(rows, columns, elements);
    }

    /*
    This method contains implementation of equations:
    hi = AF(Wxh * xi + Whh * h(i-1) + bh)
    yi = Why * hi + by
    AF is activation function, in recurrent networks it's usually tanh.
    */
    @NonNull
    public List<Vector> predict (List<Vector> inputSignals) throws VectorMatrixException {
        lastInputs = new ArrayList<>(inputSignals);
        lastHiddenValues = new ArrayList<>();
        List<Vector> result = new ArrayList<>();
        Vector h = new Vector(topology.getHiddenCount(), VectorType.COLUMN);
        lastHiddenValues.add(h);
        List<Vector> y = new ArrayList<>();
        ActivationFunction AF = topology.getActivationFunction();
        for (Vector inputSignal : inputSignals) {
            //Wxh * xi
            Vector first = Wxh.mul(inputSignal);
            //Whh * h(i-1)
            Vector second = Whh.mul(h);
            //Wxh * xi + Whh * h(i-1) + bh
            Vector res = first.add(second).add(bh);
            //hi = AF(Wxh * xi + Whh * h(i-1) + bh)
            h = AF.function(res);
            //yi = Why * hi + by
            Vector yi = Why.mul(h).add(by);
            y.add(yi);
            lastHiddenValues.add(h);
        }
        for (int i = y.size() - topology.getOutputCount(); i < y.size(); i++) {
            result.add(y.get(i));
        }
        return result;
    }

    public void learn(RNNDataset dataSet, int numberOfSteps) throws VectorMatrixException {
        double learningRate = topology.getLearningRate();
        for (int j = 0; j < numberOfSteps; j++) {
            for (int i = 0; i < dataSet.getSize(); i++) {
                List<Vector> inputs = dataSet.getInputSignals(i);
                List<Vector> outputs = dataSet.getExpectedOutputs(i);
                List<Vector> result = predict(inputs);

                ActivationFunction AF = topology.getActivationFunction();

                //first of all - calculate loss function
                Vector d_y = topology.getLossFunction().gradient(result.get(0), outputs.get(0));

                //calculate values dE/dby and dE/dWhy
                Matrix d_Why = d_y.mul(lastHiddenValues.get(inputs.size()).T());
                Vector d_by = new Vector(d_y);

                //it is special value: dE/dh_i. Last value: dE/dh_n = dE/dy *Why
                Vector d_h = Why.T().mul(d_y);

                //create matrices for gradients
                Matrix d_Whh = new Matrix(Whh.getRows(), Whh.getColumns());
                Matrix d_Wxh = new Matrix(Wxh.getRows(), Wxh.getColumns());
                Vector d_bh = new Vector(bh.size(), bh.getType());
                for (int k = inputs.size() - 1; k >= 0; k--) {
                    //Wxh * xi
                    Vector first = Wxh.mul(lastInputs.get(k));
                    //Whh * h(i-1)
                    Vector second = Whh.mul(lastHiddenValues.get(k+1));
                    //Wxh * xi + Whh * h(i-1) + bh
                    Vector res = first.add(second).add(bh);
                    //Gradient: dhi\dWxh = dAF(x)\dx * dx\dWxh, where x = Wxh * xi + Whh * h(i-1) + bh
                    //temp is value: dAF(x)\dx *dE\dh
                    Vector temp = AF.derivative(res).mulElemByElem(d_h);

                    d_bh = d_bh.add(temp);
                    d_Whh = d_Whh.add(temp.mul(lastHiddenValues.get(k+1).T()));
                    d_Wxh = d_Wxh.add(temp.mul(lastInputs.get(k).T()));
                }

                //limit the values in gradients to avoid the problem of vanishing gradients
                d_Why = d_Why.limit(-1.0, 1.0);
                d_by = d_by.limit(-1.0, 1.0);

                d_Whh = d_Whh.limit(-1.0, 1.0);
                d_Wxh = d_Wxh.limit(-1.0, 1.0);
                d_bh = d_bh.limit(-1.0, 1.0);

                //update parameters of RNN
                Why = Why.sub(d_Why.mul(learningRate));
                by = by.sub(d_by.mul(learningRate));

                Whh = Whh.sub(d_Whh.mul(learningRate));
                Wxh = Wxh.sub(d_Wxh.mul(learningRate));
                bh = bh.sub(d_bh.mul(learningRate));
            }
        }
    }
}

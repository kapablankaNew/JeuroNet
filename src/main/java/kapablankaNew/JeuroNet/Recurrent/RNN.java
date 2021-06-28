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
}

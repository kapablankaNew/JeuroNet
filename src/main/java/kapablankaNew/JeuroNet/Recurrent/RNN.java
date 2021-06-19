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

import kapablankaNew.JeuroNet.Mathematical.Matrix;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
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
}

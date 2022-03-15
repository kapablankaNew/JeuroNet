package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.Matrix;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class RnnLayer implements RecurrentLayer, Serializable {
    public RnnLayer (@NonNull RNNTopology topology) throws VectorMatrixException {
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

    @Override
    public List<Vector> predict(List<Vector> inputSignals) throws VectorMatrixException {
        lastInputs = new ArrayList<>(inputSignals);
        lastValuesH = new ArrayList<>();
        lastValuesZ = new ArrayList<>();
        List<Vector> result = new ArrayList<>();
        Vector h = new Vector(topology.getHiddenCount(), VectorType.COLUMN);
        Vector z;
        List<Vector> y = new ArrayList<>();
        ActivationFunction AF = topology.getActivationFunction();
        for (Vector inputSignal : inputSignals) {
            //Wxh * xi
            Vector first = Wxh.mul(inputSignal);
            //Whh * h(i-1)
            Vector second = Whh.mul(h);
            //Wxh * xi + Whh * h(i-1) + bh
            z = first.add(second).add(bh);
            //hi = AF(Wxh * xi + Whh * h(i-1) + bh)
            h = AF.function(z);
            //yi = Why * hi + by
            Vector yi = (Why.mul(h)).add(by);
            y.add(yi);
            lastValuesH.add(h);
            lastValuesZ.add(z);
        }
        for (int i = y.size() - topology.getOutputCount(); i < y.size(); i++) {
            result.add(y.get(i));
        }
        return result;
    }

    @Override
    public List<Vector> learn(List<Vector> inputSignals, List<Vector> errors) throws VectorMatrixException {
        predict(inputSignals);

        ActivationFunction AF = topology.getActivationFunction();
        double learningRate = topology.getLearningRate();

        //first - calculate loss function
        Vector d_y = new Vector(errors.get(0));

        //calculate values dE/dby and dE/dWhy
        Matrix d_Why = d_y.mul(lastValuesH.get(lastValuesH.size() - 1).T());
        Vector d_by = new Vector(d_y);

        //it is special value: dE/dzi
        Vector temp = Why.T().mul(d_y).mulElemByElem(AF.derivative(lastValuesZ.get(lastValuesZ.size() - 1)));

        //create matrices for gradients
        Matrix d_Whh = new Matrix(Whh.getRows(), Whh.getColumns());
        Matrix d_Wxh = new Matrix(Wxh.getRows(), Wxh.getColumns());
        Vector d_bh = new Vector(bh.size(), bh.getType());

        for (int k = inputSignals.size() - 1; k >= 0; k--) {
            Vector h_k = lastValuesH.get(k);
            Vector z_k = lastValuesZ.get(k);

            //update gradient values
            d_Whh = d_Whh.add(temp.mul(h_k.T()));
            d_Wxh = d_Wxh.add(temp.mul(lastInputs.get(k).T()));
            d_bh = d_bh.add(temp);

            //dE/dzi = Whh.T * dE/dz(i+1) * dh(i+1)/dzi
            temp = Whh.T().mul(temp.mulElemByElem(AF.derivative(z_k)));
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
        return null;
    }

    private static Matrix createWeightsMatrix(int rows, int columns) throws VectorMatrixException {
        List<List<Double>> elements = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                row.add(ThreadLocalRandom.current().nextDouble(0.0, 0.1));
            }
            elements.add(row);
        }
        return new Matrix(rows, columns, elements);
    }

    @Getter
    private final RNNTopology topology;

    private Matrix Wxh;

    private Matrix Whh;

    private Matrix Why;

    private Vector bh;

    private Vector by;

    //Next three fields storage data about last feed forward step.
    //These data use in the learning process
    private List<Vector> lastInputs;

    private List<Vector> lastValuesH;

    private List<Vector> lastValuesZ;

}

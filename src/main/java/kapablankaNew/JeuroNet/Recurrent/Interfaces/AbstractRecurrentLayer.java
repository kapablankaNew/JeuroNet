package kapablankaNew.JeuroNet.Recurrent.Interfaces;

import kapablankaNew.JeuroNet.Mathematical.Matrix;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Recurrent.RecurrentLayerType;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

@EqualsAndHashCode
public abstract class AbstractRecurrentLayer implements RecurrentLayer, Serializable {
    @Getter
    protected final RecurrentLayerTopology topology;

    @EqualsAndHashCode.Exclude
    protected List<Vector> lastInputs;

    @EqualsAndHashCode.Exclude
    protected List<Vector> lastOutputs;

    protected AbstractRecurrentLayer(RecurrentLayerTopology topology) {
        this.topology = topology;
        lastInputs = new ArrayList<>();
        lastOutputs = new ArrayList<>();
    }

    @Override
    public abstract List<Vector> predict(List<Vector> inputSignals) throws VectorMatrixException;

    @Override
    public abstract List<Vector> learn(List<Vector> inputSignals, List<Vector> errorsGradients) throws VectorMatrixException;

    protected void updateInputs() {
        RecurrentLayerType layerType = topology.getRecurrentLayerType();
        int outputCount = topology.getOutputCount();
        int inputCount = lastInputs.size();
        int inputSize = topology.getInputSize();
        if (layerType == RecurrentLayerType.NO_INPUT) {
            while(lastInputs.size() < outputCount) {
                lastInputs.add(new Vector(inputSize));
            }
        } else if (layerType == RecurrentLayerType.NO_INPUT_NO_OUTPUT) {
            while (lastInputs.size() < outputCount + inputCount - 1) {
                lastInputs.add(new Vector(inputSize));
            }
        }
    }

    protected synchronized List<Vector> updateInputs(List<Vector> inputs) {
        RecurrentLayerType layerType = topology.getRecurrentLayerType();
        List<Vector> result = new ArrayList<>(inputs);
        int outputCount = topology.getOutputCount();
        int inputCount = inputs.size();
        int inputSize = topology.getInputSize();
        if (layerType == RecurrentLayerType.NO_INPUT) {
            while(result.size() < outputCount) {
                result.add(new Vector(inputSize));
            }
        } else if (layerType == RecurrentLayerType.NO_INPUT_NO_OUTPUT) {
            while (result.size() < outputCount + inputCount - 1) {
                result.add(new Vector(inputSize));
            }
        }
        return result;
    }

    protected synchronized void updateOutputs() {
        RecurrentLayerType layerType = topology.getRecurrentLayerType();
        int outputCount = topology.getOutputCount();
        if (layerType == RecurrentLayerType.NO_OUTPUT || layerType == RecurrentLayerType.NO_INPUT_NO_OUTPUT) {
            while (lastOutputs.size() > outputCount) {
                lastOutputs.remove(0);
            }
        }
    }

    protected List<Vector> updateOutputs(List<Vector> outputs) {
        List<Vector> result = new ArrayList<>(outputs);
        RecurrentLayerType layerType = topology.getRecurrentLayerType();
        int outputCount = topology.getOutputCount();
        if (layerType == RecurrentLayerType.NO_OUTPUT || layerType == RecurrentLayerType.NO_INPUT_NO_OUTPUT) {
            while (result.size() > outputCount) {
                result.remove(0);
            }
        }
        return result;
    }

    protected Matrix createWeightsMatrix(int rows, int columns) throws VectorMatrixException {
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
}

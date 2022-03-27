package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractRecurrentLayer implements RecurrentLayer {
    @Getter
    protected final RnnLayerTopology topology;

    protected List<Vector> lastInputs;

    protected List<Vector> lastOutputs;

    protected AbstractRecurrentLayer(RnnLayerTopology topology) {
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

    protected void updateOutputs() {
        RecurrentLayerType layerType = topology.getRecurrentLayerType();
        int outputCount = topology.getOutputCount();
        if (layerType == RecurrentLayerType.NO_OUTPUT || layerType == RecurrentLayerType.NO_INPUT_NO_OUTPUT) {
            while (lastOutputs.size() > outputCount) {
                lastOutputs.remove(0);
            }
        }
    }
}

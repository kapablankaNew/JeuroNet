package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.DataSetException;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Storable;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode
public class RecurrentDataset implements Storable {
    private final List<List<Vector>> inputSignals;

    private final List<List<Vector>> expectedOutputs;

    @Getter
    private final int inputSize;

    @Getter
    private final int outputCount;

    @Getter
    private final int outputSize;

    public RecurrentDataset(int inputSize, int outputCount, int outputSize) throws DataSetException {
        if (outputCount <= 0) {
            throw new DataSetException("Number of outputs must be greater than 0!");
        }
        if (inputSize <= 0) {
            throw new DataSetException("Size of the input data must be greater than 0!");
        }
        if (outputSize <= 0) {
            throw new DataSetException("Size of the output data must be greater than 0!");
        }

        this.inputSize = inputSize;
        this.outputCount = outputCount;
        this.outputSize = outputSize;

        inputSignals = new ArrayList<>();
        expectedOutputs = new ArrayList<>();
    }

    public int getSize() {
        return inputSignals.size();
    }

    public List<Vector> getInputSignals(int index) {
        return inputSignals.get(index);
    }

    public List<Vector> getExpectedOutputs(int index) {
        return expectedOutputs.get(index);
    }

    public void addData(List<Vector> inputs, List<Vector> outputs) throws DataSetException {
        for (Vector input : inputs) {
            if (input.size() != getInputSize()) {
                throw new DataSetException("Number of input signals is not equal to input size of dataset!");
            }
        }
        if (outputs.size() != getOutputCount()) {
            throw new DataSetException("Number of output vectors is not equal to output count of dataset!");
        }
        for (Vector output : outputs) {
            if (output.size() != getOutputSize()) {
                throw new DataSetException("Number of output signals is not equal to output size of dataset!");
            }
        }

        List<Vector> ins = new ArrayList<>();
        for (Vector input : inputs) {
            Vector vector = new Vector(input);
            ins.add(vector);
        }
        inputSignals.add(ins);

        List<Vector> outs = new ArrayList<>();
        for (Vector output : outputs) {
            Vector vector = new Vector(output);
            outs.add(vector);
        }
        expectedOutputs.add(outs);
    }

    @Override
    public void save(String path) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(path);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(this);
        objectOutputStream.close();
        fileOutputStream.close();
    }

    public static RecurrentDataset load(String path) throws IOException, ClassNotFoundException {
        FileInputStream fileInputStream = new FileInputStream(path);
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        RecurrentDataset dataset = (RecurrentDataset) objectInputStream.readObject();
        objectInputStream.close();
        fileInputStream.close();
        return dataset;
    }
}

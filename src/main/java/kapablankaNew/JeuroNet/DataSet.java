package kapablankaNew.JeuroNet;

import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class DataSet implements Serializable {
    private final List<List<Double>> inputSignals;

    private final List<List<Double>> expectedResults;

    @Getter
    private final int outputCount;

    @Getter
    private final int inputCount;

    private final List<Double> avr;

    private final List<Double> standardDeviation;

    private final List<Double> max;

    private final List<Double> min;

    public DataSet(int inputCount, int outputCount) throws DataSetException {
        if (inputCount <= 0) {
            throw new DataSetException("Number of inputs must be greater than 0");
        }
        if (outputCount <= 0) {
            throw new DataSetException("Number of outputs must be greater than 0");
        }
        inputSignals = new ArrayList<>();
        expectedResults = new ArrayList<>();
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        avr = new ArrayList<>();
        standardDeviation = new ArrayList<>();
        min = new ArrayList<>();
        max = new ArrayList<>();
    }

    @NonNull
    public List<Double> getInputSignals(int index) {
        return inputSignals.get(index);
    }

    @NonNull
    public List<Double> getExpectedResult(int index) {
        return expectedResults.get(index);
    }

    public int getSize() {
        return inputSignals.size();
    }

    public void addData(List<Double> inputs, List<Double> results) throws DataSetException {
        if (inputs.size() != inputCount) {
            throw new DataSetException("Number of input signals is not equal to input count of dataset");
        }
        if (results.size() != outputCount) {
            throw new DataSetException("Number of output results is not equal to output count of dataset");
        }
        inputSignals.add(inputs);
        expectedResults.add(results);
    }

    public void normalize() {
        int n = inputSignals.size();
        avr.clear();
        standardDeviation.clear();
        for (int i = 0; i < inputCount; i++) {
            double sum = 0;
            for (List<Double> entry : inputSignals) {
                sum += entry.get(i);
            }
            avr.add(sum / n);
            sum = 0;
            for (List<Double> entry : inputSignals) {
                sum += Math.pow(entry.get(i) - avr.get(i), 2);
            }
            standardDeviation.add(Math.sqrt(sum / n));
            for (List<Double> entry : inputSignals) {
                entry.set(i, (entry.get(i) - avr.get(i)) / standardDeviation.get(i));
            }
        }
    }

    public void normalizeEntry(List<Double> inputSignals) throws DataSetException {
        if (inputSignals.size() != inputCount) {
            throw new DataSetException("Number of input signals is not equal to input count of dataset");
        }
        for (int i = 0; i < inputCount; i++) {
            inputSignals.set(i, (inputSignals.get(i) - avr.get(i)) / standardDeviation.get(i));
        }
    }

    public void scale() {
        max.clear();
        min.clear();
        for (int i = 0; i < inputCount; i++) {
            double maximum, minimum;
            maximum = minimum = inputSignals.get(0).get(i);
            for (List<Double> entry : inputSignals) {
                double item = entry.get(i);
                if (item > maximum) {
                    maximum = item;
                }
                if (item < minimum) {
                    minimum = item;
                }
            }
            max.add(maximum);
            min.add(minimum);
            for (List<Double> entry : inputSignals) {
                entry.set(i, (entry.get(i) - min.get(i)) / (max.get(i) - min.get(i)));
            }
        }
    }

    public void scaleEntry(List<Double> inputSignals) throws DataSetException {
        if (inputSignals.size() != inputCount) {
            throw new DataSetException("Number of input signals is not equal to input count of dataset");
        }
        for (int i = 0; i < inputCount; i++) {
            inputSignals.set(i, (inputSignals.get(i) - min.get(i)) / (max.get(i) - min.get(i)));
        }
    }
}

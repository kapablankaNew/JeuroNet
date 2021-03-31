package kapablankaNew.JeuroNet.Mathematical;

import java.util.List;
import java.util.stream.IntStream;

public enum LossFunction {
    MAE {
        @Override
        public double loss(List<Double> actual, List<Double> expected) {
            return (IntStream.range(0, actual.size() - 1).
                    mapToDouble(i -> Math.abs(actual.get(i) - expected.get(i))).
                    sum()) / actual.size();
        }
    },

    MSE {
        @Override
        public double loss(List<Double> actual, List<Double> expected) {
            return (IntStream.range(0, actual.size() - 1).
                    mapToDouble(i -> Math.pow(actual.get(i) - expected.get(i), 2)).
                    sum()) / actual.size();
        }
    },

    //this metric work only for classification!
    CrossEntropy {
        @Override
        public double loss(List<Double> actual, List<Double> expected) {
            return (IntStream.range(0, actual.size() - 1).
                    mapToDouble(i -> -expected.get(i) * Math.log(actual.get(i)) -
                            (1 - expected.get(i)) * (Math.log(1 - actual.get(i)))).
                    sum()) / actual.size();
        }
    };

    public abstract double loss(List<Double> actual, List<Double> expected);
}

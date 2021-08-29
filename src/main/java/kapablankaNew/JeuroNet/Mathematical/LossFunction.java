package kapablankaNew.JeuroNet.Mathematical;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public enum LossFunction implements Serializable {
    MAE {
        @Override
        public double loss(List<Double> actual, List<Double> expected) {
            return (IntStream.range(0, actual.size()).
                    mapToDouble(i -> Math.abs(actual.get(i) - expected.get(i))).
                    sum()) / actual.size();
        }

        @Override
        public List<Double> gradient(List<Double> actual, List<Double> expected) {
            /*
            dE/dyi = (1/m) * ((yi - y~i)/abs(yi - y~i))
            yi - real value, y~i - expected value
            ((yi - y~i)/abs(yi - y~i)): if real > expected - it is 1,
            if real < expected - it is -1, if real = expected - it is 0
            */
            return IntStream.range(0, actual.size()).
                    mapToDouble(i -> actual.get(i).compareTo(expected.get(i))).
                    mapToObj(i -> i/actual.size()).
                    collect(Collectors.toList());
        }
    },

    MSE {
        @Override
        public double loss(List<Double> actual, List<Double> expected) {
            return (IntStream.range(0, actual.size()).
                    mapToDouble(i -> Math.pow(actual.get(i) - expected.get(i), 2)).
                    sum()) / actual.size();
        }

        @Override
        public List<Double> gradient(List<Double> actual, List<Double> expected) {
            /*
            dE/dyi = (1/m) * 2 * (yi - y~i)
            yi - real value, y~i - expected value
            */
            return IntStream.range(0, actual.size()).
                    mapToObj(i -> 2 * (actual.get(i) - expected.get(i))).
                    map(i -> i/actual.size()).
                    collect(Collectors.toList());
        }
    },

    //this metric work only for classification!
    CrossEntropy {
        @Override
        public double loss(List<Double> actual, List<Double> expected) {
            return (IntStream.range(0, actual.size()).
                    mapToDouble(i -> -expected.get(i) * Math.log(actual.get(i)) -
                            (1 - expected.get(i)) * (Math.log(1 - actual.get(i)))).
                    sum()) / actual.size();
        }

        @Override
        public List<Double> gradient(List<Double> actual, List<Double> expected) {
            return IntStream.range(0, actual.size()).
                    mapToDouble(i -> {
                        if (expected.get(i) != 0.0 && expected.get(i) != 1.0)
                            throw new ArithmeticException("Expected values should be 0 or 1!");
                        if (expected.get(i) == 1.0)
                            return -expected.get(i)/actual.get(i);
                        return -(1 - expected.get(i))/ (actual.get(i) - 1);
                    }).mapToObj(i -> i / actual.size()).
                    collect(Collectors.toList());
        }
    };

    public abstract double loss(List<Double> actual, List<Double> expected);

    public double loss(Vector actual, Vector expected) {
        List<Double> act = actual.getElements();
        List<Double> exp = expected.getElements();
        return this.loss(act, exp);
    }

    public abstract List<Double> gradient(List<Double> actual, List<Double> expected);

    public Vector gradient (Vector actual, Vector expected) throws VectorMatrixException {
        if (actual.getType() != expected.getType()) {
            throw new VectorMatrixException("It's not possible to calculate gradient for row and column!");
        }
        List<Double> act = actual.getElements();
        List<Double> exp = expected.getElements();
        return new Vector(this.gradient(act, exp), actual.getType());
    }
}

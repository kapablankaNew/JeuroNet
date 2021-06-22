package kapablankaNew.JeuroNet.Mathematical;

import java.util.ArrayList;
import java.util.List;

public enum ActivationFunction {
    SIGMOID {
        @Override
        public double function(double x) {
            return 1/(1 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            return function(x)*(1 - function(x));
        }
    },
    TANH {
        @Override
        public double function(double x) {
            return (Math.exp(2*x)-1)/(Math.exp(2*x)+1);
        }

        @Override
        public double derivative(double x) {
            return 1 - Math.pow(function(x), 2);
        }
    },
    RELU {
        @Override
        public double function(double x) {
            return x < 0 ? 0 : x;
        }

        @Override
        public double derivative(double x) {
            return x < 0 ? 0 : 1;
        }
    },
    LINEAR {
        @Override
        public double function(double x) {
            return x;
        }

        @Override
        public double derivative(double x) {
            return 1;
        }
    };
    public abstract double function(double x);

    public Vector function(Vector x) {
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            result.add(function(x.get(i)));
        }
        return new Vector(result, x.getType());
    }

    public abstract double derivative(double x);

    public Vector derivative(Vector x) {
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            result.add(derivative(x.get(i)));
        }
        return new Vector(result, x.getType());
    }
}

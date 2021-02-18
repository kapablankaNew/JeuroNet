package kapablankaNew.JeuroNet.Mathematical;

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

    public abstract double derivative(double x);
}

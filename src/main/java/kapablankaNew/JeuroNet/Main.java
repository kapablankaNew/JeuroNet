package kapablankaNew.JeuroNet;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.LossFunction;
import kapablankaNew.JeuroNet.Recurrent.RnnLayerTopology;

public class Main {
    public static void main(String[] args) throws TopologyException {
        RnnLayerTopology topology = RnnLayerTopology.builder()
                .outputCount(-1)
                .hiddenCount(10)
                .outputSize(5)
                .inputSize(1)
                .learningRate(0.01)
                .activationFunction(ActivationFunction.TANH)
                .lossFunction(LossFunction.MSE)
                .build();
    }
}

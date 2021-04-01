package kapablankaNew.JeuroNet.MLP;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import lombok.Getter;

public class LayerInfo {
    @Getter
    private final int numberOfNeurons;

    @Getter
    private final ActivationFunction activationFunction;

    public LayerInfo(int numberOfNeurons, ActivationFunction activationFunction) {
        this.numberOfNeurons = numberOfNeurons;
        this.activationFunction = activationFunction;
    }
}

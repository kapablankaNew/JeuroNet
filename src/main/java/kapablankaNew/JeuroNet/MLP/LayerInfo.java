package kapablankaNew.JeuroNet.MLP;

import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;

@EqualsAndHashCode
public class LayerInfo implements Serializable {
    @Getter
    private final int numberOfNeurons;

    @Getter
    private final ActivationFunction activationFunction;

    public LayerInfo(int numberOfNeurons, ActivationFunction activationFunction) {
        this.numberOfNeurons = numberOfNeurons;
        this.activationFunction = activationFunction;
    }
}

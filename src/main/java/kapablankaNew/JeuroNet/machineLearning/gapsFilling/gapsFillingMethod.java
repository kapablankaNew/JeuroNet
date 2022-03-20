package kapablankaNew.JeuroNet.machineLearning.gapsFilling;

import kapablankaNew.JeuroNet.Mathematical.Vector;

public interface gapsFillingMethod {
    void train();

    Vector predict();
}

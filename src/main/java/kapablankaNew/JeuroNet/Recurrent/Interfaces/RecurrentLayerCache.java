package kapablankaNew.JeuroNet.Recurrent.Interfaces;

import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Recurrent.RecurrentLayerType;
import lombok.Getter;

import java.util.List;

abstract class RecurrentLayerCache {
    @Getter
    protected List<Vector> lastInputs, lastOutputs;

    void updateInputs(RecurrentLayerType type) {

    }

    void updateOutputs(RecurrentLayerType type) {

    }
}

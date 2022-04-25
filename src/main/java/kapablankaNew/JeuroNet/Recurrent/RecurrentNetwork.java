package kapablankaNew.JeuroNet.Recurrent;

import kapablankaNew.JeuroNet.Mathematical.LossFunction;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Recurrent.Interfaces.RecurrentLayer;
import kapablankaNew.JeuroNet.Recurrent.Interfaces.RecurrentLayerTopology;
import kapablankaNew.JeuroNet.Storable;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode
public class RecurrentNetwork implements Storable {
    private final List<RecurrentLayer> layers;

    @Getter
    private final LossFunction lossFunction;

    private List<List<Vector>> lastLayersInputs;

    public RecurrentNetwork(List<RecurrentLayerTopology> topologies, LossFunction lossFunction)
            throws VectorMatrixException {
        this.lossFunction = lossFunction;
        layers = new ArrayList<>();
        for (var topology: topologies) {
            RecurrentLayer layer = topology.createRecurrentLayer();
            layers.add(layer);
        }
    }

    @NonNull
    public List<Vector> predict (List<Vector> inputSignals) throws VectorMatrixException {
        lastLayersInputs = new ArrayList<>();
        List<Vector> result = new ArrayList<>(inputSignals);
        lastLayersInputs.add(new ArrayList<>(result));
        for(var layer: layers) {
            result = layer.predict(result);
            lastLayersInputs.add(new ArrayList<>(result));
        }
        lastLayersInputs.remove(lastLayersInputs.size() - 1);
        return result;
    }

    public void learn(RecurrentDataset dataSet, int numberOfSteps) throws VectorMatrixException {
        for (int j = 0; j < numberOfSteps; j++) {
            for (int i = 0; i < dataSet.getSize(); i++) {
                List<Vector> inputs = dataSet.getInputSignals(i);
                List<Vector> expectedOutputs = dataSet.getExpectedOutputs(i);
                List<Vector> result = predict(inputs);

                if (result.size() != expectedOutputs.size()) {
                    throw new VectorMatrixException("Error in recurrent network! Expected output size = " +
                            expectedOutputs.size() + ", real output size = " + result.size());
                }
                //first - calculate loss function
                List<Vector> errorsGradients = new ArrayList<>();
                for (int k = 0; k < expectedOutputs.size(); k++) {
                    Vector errorGradient = lossFunction.gradient(result.get(k), expectedOutputs.get(k));
                    errorsGradients.add(errorGradient);
                }
                for (int k = layers.size() - 1; k >= 0; k--) {
                    errorsGradients = layers.get(k).learn(lastLayersInputs.get(k), errorsGradients);
                }
            }
        }
    }

    @Override
    public void save(String path) throws IOException {
        if (! path.endsWith(".jnn")) {
            throw new IOException("Incorrect filename! File format must be '.jnn'!");
        }
        FileOutputStream fileOutputStream = new FileOutputStream(path);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(this);
        objectOutputStream.close();
        fileOutputStream.close();
    }

    public static RecurrentNetwork load(String path) throws IOException, ClassNotFoundException {
        if (! path.endsWith(".jnn")) {
            throw new IOException("Incorrect filename! File format must be '.jnn'!");
        }
        FileInputStream fileInputStream = new FileInputStream(path);
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        RecurrentNetwork network = (RecurrentNetwork) objectInputStream.readObject();
        objectInputStream.close();
        fileInputStream.close();
        return network;
    }
}

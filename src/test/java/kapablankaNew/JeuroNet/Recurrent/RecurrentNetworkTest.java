package kapablankaNew.JeuroNet.Recurrent;

import com.github.cliftonlabs.json_simple.JsonException;
import com.github.cliftonlabs.json_simple.JsonObject;
import com.github.cliftonlabs.json_simple.Jsoner;
import kapablankaNew.JeuroNet.DataSetException;
import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.LossFunction;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
import kapablankaNew.JeuroNet.Recurrent.GRU.GruLayerTopology;
import kapablankaNew.JeuroNet.Recurrent.Interfaces.RecurrentLayerTopology;
import kapablankaNew.JeuroNet.Recurrent.LSTM.LstmLayerTopology;
import kapablankaNew.JeuroNet.Recurrent.RNN.RnnLayerTopology;
import kapablankaNew.JeuroNet.TextConverter;
import kapablankaNew.JeuroNet.TopologyException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class RecurrentNetworkTest {
    private static TextConverter converter;
    private static RecurrentDataset train;
    private static RecurrentDataset test;

    @BeforeAll
    static void setup() throws JsonException, DataSetException, IOException {
        train = getDatasetFromFile("src/test/resources/Train.json");
        test = getDatasetFromFile("src/test/resources/Test.json");
    }

    @Test
    public void learnRnnTest() throws VectorMatrixException, TopologyException {
        RnnLayerTopology topology = RnnLayerTopology.builder()
                .inputSize(converter.getNumberUniqueWords())
                .outputCount(1)
                .outputSize(2)
                .hiddenSize(100)
                .learningRate(0.0001)
                .activationFunction(ActivationFunction.TANH)
                .recurrentLayerType(RecurrentLayerType.NO_OUTPUT)
                .build();

        List<RecurrentLayerTopology> topologies = List.of(topology);
        RecurrentNetwork recurrentNetwork = new RecurrentNetwork(topologies, LossFunction.MAE);
        recurrentNetwork.learn(train, 50);
        double loss = calcLoss(recurrentNetwork, test);

        assertTrue(loss < 0.5);
    }

    @Test
    public void learnGruTest() throws VectorMatrixException, TopologyException {
        GruLayerTopology topology = GruLayerTopology.builder()
                .inputSize(converter.getNumberUniqueWords())
                .outputCount(20)
                .outputSize(25)
                .hiddenSize(20)
                .learningRate(0.001)
                .recurrentLayerType(RecurrentLayerType.NO_INPUT)
                .build();

        GruLayerTopology topology1 = GruLayerTopology.builder()
                .inputSize(25)
                .outputCount(1)
                .outputSize(2)
                .hiddenSize(20)
                .learningRate(0.001)
                .recurrentLayerType(RecurrentLayerType.NO_OUTPUT)
                .build();

        List<RecurrentLayerTopology> topologies = List.of(topology, topology1);
        RecurrentNetwork recurrentNetwork = new RecurrentNetwork(topologies, LossFunction.MAE);
        recurrentNetwork.learn(train, 150);
        double loss = calcLoss(recurrentNetwork, test);
        System.out.println(loss);
        assertTrue(loss < 0.5);
    }

    @Test
    public void learnLstmTest() throws VectorMatrixException, TopologyException {
        LstmLayerTopology topology = LstmLayerTopology.builder()
                .inputSize(converter.getNumberUniqueWords())
                .outputCount(20)
                .outputSize(25)
                .hiddenSize(20)
                .learningRate(0.001)
                .recurrentLayerType(RecurrentLayerType.NO_INPUT)
                .build();

        LstmLayerTopology topology1 = LstmLayerTopology.builder()
                .inputSize(25)
                .outputCount(1)
                .outputSize(2)
                .hiddenSize(20)
                .learningRate(0.001)
                .recurrentLayerType(RecurrentLayerType.NO_OUTPUT)
                .build();

        List<RecurrentLayerTopology> topologies = List.of(topology, topology1);
        RecurrentNetwork recurrentNetwork = new RecurrentNetwork(topologies, LossFunction.MAE);
        recurrentNetwork.learn(train, 150);
        double loss = calcLoss(recurrentNetwork, test);
        assertTrue(loss < 0.5);
    }

    @Test
    public void saveLoadTest() throws TopologyException, VectorMatrixException, IOException, ClassNotFoundException {
        RnnLayerTopology topology = RnnLayerTopology.builder()
                .inputSize(converter.getNumberUniqueWords())
                .outputCount(20)
                .outputSize(10)
                .hiddenSize(10)
                .learningRate(0.0001)
                .activationFunction(ActivationFunction.TANH)
                .recurrentLayerType(RecurrentLayerType.NO_INPUT)
                .build();

        GruLayerTopology topology1 = GruLayerTopology.builder()
                .inputSize(10)
                .outputCount(20)
                .outputSize(5)
                .hiddenSize(10)
                .learningRate(0.001)
                .recurrentLayerType(RecurrentLayerType.ALL_INPUT_ALL_OUTPUT)
                .build();

        LstmLayerTopology topology2 = LstmLayerTopology.builder()
                .inputSize(5)
                .outputCount(1)
                .outputSize(2)
                .hiddenSize(5)
                .learningRate(0.001)
                .recurrentLayerType(RecurrentLayerType.NO_OUTPUT)
                .build();

        List<RecurrentLayerTopology> topologies = List.of(topology, topology1, topology2);
        RecurrentNetwork recurrentNetwork = new RecurrentNetwork(topologies, LossFunction.MAE);
        recurrentNetwork.learn(train, 150);
        recurrentNetwork.save("Recurrent.jnn");
        RecurrentNetwork recurrentNetwork1 = RecurrentNetwork.load("Recurrent.jnn");
        assertEquals(recurrentNetwork, recurrentNetwork1);
    }

    public static RecurrentDataset getDatasetFromFile(String filename) throws DataSetException, IOException,
            JsonException {
        RecurrentDataset result;
        try (Reader JsonReader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)))) {
            JsonObject parser = (JsonObject) Jsoner.deserialize(JsonReader);
            Set<String> keys = parser.keySet();
            if (converter == null) {
                List<String> data = new ArrayList<>(keys);
                converter = new TextConverter(data);
            }
            result = new RecurrentDataset(18, 1, 2);
            for (String key : keys) {
                boolean output = (boolean) parser.get(key);
                List<Vector> expOut = new ArrayList<>();
                if (output) {
                    expOut.add(new Vector(Arrays.asList(1.0, 0.0), VectorType.COLUMN));
                } else {
                    expOut.add(new Vector(Arrays.asList(0.0, 1.0), VectorType.COLUMN));
                }
                List<Vector> inputs = converter.convert(key);
                result.addData(inputs, expOut);
            }
        }
        return result;
    }

    public double calcLoss(RecurrentNetwork network, RecurrentDataset dataset) throws VectorMatrixException {
        double loss = 0.0;
        for (int j = 0; j < dataset.getSize(); j++) {
            List<Vector> ins = dataset.getInputSignals(j);
            List<Vector> expOut = dataset.getExpectedOutputs(j);
            List<Vector> result = network.predict(ins);
            for (int i = 0; i < result.size(); i++) {
                Vector res = result.get(i);
                Vector exp = expOut.get(i);
                loss += network.getLossFunction().loss(res, exp);
            }
        }
        loss /= dataset.getSize();

        return loss;
    }
}
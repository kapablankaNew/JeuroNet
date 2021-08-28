package kapablankaNew.JeuroNet.Recurrent;

import com.github.cliftonlabs.json_simple.JsonException;
import com.github.cliftonlabs.json_simple.JsonObject;
import com.github.cliftonlabs.json_simple.Jsoner;
import kapablankaNew.JeuroNet.DataSetException;
import kapablankaNew.JeuroNet.Mathematical.*;
import kapablankaNew.JeuroNet.TextConverter;
import kapablankaNew.JeuroNet.TopologyException;
import org.junit.Assert;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class RecurrentNetworkTest {
    private TextConverter converter;
    @Test
    public void predictTest() throws IOException, JsonException, TopologyException,
            VectorMatrixException, DataSetException {
        converter = null;
        RecurrentDataset train = getDatasetFromFile("src/test/resources/Train.json");
        RecurrentDataset test = getDatasetFromFile("src/test/resources/Test.json");
        RNNTopology topology = new RNNTopology(converter.getNumberUniqueWords(),
                        1, 2, 100, 0.0001,
                ActivationFunction.TANH, LossFunction.MAE);

        RecurrentNetwork network = new RecurrentNetwork(topology);
        network.learn(train, 50);

        double loss = calcLoss(network, test);
        Assert.assertTrue(loss < 0.5);
    }

    public RecurrentDataset getDatasetFromFile(String filename) throws DataSetException, IOException,
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

    public static double calcLoss(RecurrentNetwork network, RecurrentDataset dataset) throws VectorMatrixException {
        RNNTopology topology = network.getTopology();
        double loss = 0.0;
        for (int j = 0; j < dataset.getSize(); j++) {
            List<Vector> ins = dataset.getInputSignals(j);
            List<Vector> expOut = dataset.getExpectedOutputs(j);
            Vector res = network.predict(ins).get(0);
            Vector exp = expOut.get(0);
            loss += topology.getLossFunction().loss(res, exp);
        }
        loss /= dataset.getSize();

        return loss;
    }
}
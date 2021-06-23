package kapablankaNew.JeuroNet.Recurrent;

import com.github.cliftonlabs.json_simple.JsonException;
import com.github.cliftonlabs.json_simple.JsonKey;
import com.github.cliftonlabs.json_simple.JsonObject;
import com.github.cliftonlabs.json_simple.Jsoner;
import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorMatrixException;
import kapablankaNew.JeuroNet.TextConverter;
import kapablankaNew.JeuroNet.TopologyException;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.*;

public class RNNTest {

    @Test
    public void predictTest() throws IOException, JsonException, TopologyException,
            VectorMatrixException {
        String filename = "src/test/resources/Train.json";
        TextConverter converter;
        try (Reader JsonReader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)))) {
            JsonObject parser = (JsonObject) Jsoner.deserialize(JsonReader);
            Set<String> keys = parser.keySet();
            List<String> data = new ArrayList<>(keys);
            converter = new TextConverter(data);
        }
        List<Vector> ins = converter.convert("i am very good");
        RNNTopology topology = new RNNTopology(converter.getNumberUniqueWords(),
                1, 2, 64, ActivationFunction.TANH);
        RNN network = new RNN(topology);
        List<Vector> res = network.predict(ins);
    }
}
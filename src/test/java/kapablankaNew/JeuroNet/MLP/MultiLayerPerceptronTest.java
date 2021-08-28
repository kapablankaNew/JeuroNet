package kapablankaNew.JeuroNet.MLP;

import kapablankaNew.JeuroNet.DataSet;
import kapablankaNew.JeuroNet.DataSetException;
import kapablankaNew.JeuroNet.Mathematical.ActivationFunction;
import kapablankaNew.JeuroNet.TopologyException;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

public class MultiLayerPerceptronTest {

    @Test
    public void learnTest() throws DataSetException, TopologyException, MultiLayerPerceptronException {
        //In this test created neural network for logical operation XOR
        DataSet dataSet = new DataSet(2, 1);
        //created dataset
        dataSet.addData(Arrays.asList(0.0, 0.0), Collections.singletonList(0.0));
        dataSet.addData(Arrays.asList(0.0, 1.0), Collections.singletonList(1.0));
        dataSet.addData(Arrays.asList(1.0, 0.0), Collections.singletonList(1.0));
        dataSet.addData(Arrays.asList(1.0, 1.0), Collections.singletonList(0.0));

        LayerInfo outLayer = new LayerInfo(1, ActivationFunction.SIGMOID);
        LayerInfo hiddenLayerOne = new LayerInfo(4, ActivationFunction.SIGMOID);
        LayerInfo hiddenLayerTwo = new LayerInfo(2, ActivationFunction.SIGMOID);
        //created topology with two hidden layers with 4 and 2 neurons
        Topology topology = new Topology(2, outLayer, Arrays.asList(hiddenLayerOne,
                hiddenLayerTwo), 0.1);

        //created NN
        MultiLayerPerceptron NN = new MultiLayerPerceptron(topology);

        //learn NN
        NN.learn(dataSet, 1_000_000);

        //check results
        Assert.assertTrue(NN.predict(Arrays.asList(0.0, 0.0)).get(0) < 0.1);
        Assert.assertTrue(NN.predict(Arrays.asList(0.0, 1.0)).get(0) > 0.9);
        Assert.assertTrue(NN.predict(Arrays.asList(1.0, 0.0)).get(0) > 0.9);
        Assert.assertTrue(NN.predict(Arrays.asList(1.0, 1.0)).get(0) < 0.1);
    }

    @Test
    public void saveLoadTest() throws TopologyException, DataSetException, MultiLayerPerceptronException,
            IOException, ClassNotFoundException {
        Topology topology = new Topology(2, new LayerInfo(2, ActivationFunction.SIGMOID),
                Collections.singletonList(new LayerInfo(2, ActivationFunction.SIGMOID)),
                0.01);
        DataSet dataSet = new DataSet(2, 2);
        dataSet.addData(Arrays.asList(1.0, 2.0), Arrays.asList(0.0, 1.0));
        MultiLayerPerceptron NN = new MultiLayerPerceptron(topology);
        NN.learn(dataSet, 2);
        NN.save("TestNN");
        MultiLayerPerceptron NnLoaded = MultiLayerPerceptron.load("TestNN");
        Assert.assertEquals(NN, NnLoaded);
    }
}
package kapablankaNew.JeuroNet.MLP;

import kapablankaNew.JeuroNet.DataSet;
import kapablankaNew.JeuroNet.DataSetException;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.*;

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

        //created topology with two hidden layers with 4 and 2 neurons
        Topology topology = new Topology(2, 1, new int[] {4, 2}, 0.01);

        //created NN
        MultiLayerPerceptron NN = new MultiLayerPerceptron(topology);

        //learn NN
        NN.learnBackPropagation(dataSet, 500_000);

        //check results
        Assert.assertTrue(NN.predict(Arrays.asList(0.0, 0.0)).get(0) < 0.05);
        Assert.assertTrue(NN.predict(Arrays.asList(0.0, 1.0)).get(0) > 0.95);
        Assert.assertTrue(NN.predict(Arrays.asList(1.0, 0.0)).get(0) > 0.95);
        Assert.assertTrue(NN.predict(Arrays.asList(1.0, 1.0)).get(0) < 0.05);
    }
}
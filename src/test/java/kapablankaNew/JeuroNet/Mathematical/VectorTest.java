package kapablankaNew.JeuroNet.Mathematical;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class VectorTest {

    @Test
    public void addTest() throws VectorMatrixException {
        Vector v1 = new Vector(Arrays.asList(1.0, 2.0, 3.0));
        Vector v2 = new Vector(Arrays.asList(1.0, 2.0, 3.0));
        Vector res = v1.add(v2);
        Vector expected = new Vector(Arrays.asList(2.0, 4.0, 6.0));
        Assert.assertEquals(expected, res);
    }

    @Test
    public void subTest() throws VectorMatrixException {
        Vector v1 = new Vector(Arrays.asList(1.0, 2.0, 3.0));
        Vector v2 = new Vector(Arrays.asList(2.0, 1.0, 1.0));
        Vector res = v1.sub(v2);
        Vector expected = new Vector(Arrays.asList(-1.0, 1.0, 2.0));
        Assert.assertEquals(expected, res);
    }
}
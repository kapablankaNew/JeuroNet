package kapablankaNew.JeuroNet.Mathematical;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

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

    @Test
    public void mulTest() throws VectorMatrixException {
        Vector v1 = new Vector(Arrays.asList(1.0, 2.0, 3.0), VectorType.ROW);
        Vector v2 = new Vector(Arrays.asList(4.0, 5.0, 6.0), VectorType.COLUMN);
        Matrix res = v1.mul(v2);
        Matrix expected = new Matrix(1, 1,
                Collections.singletonList(Collections.singletonList(32.0)));

        Assert.assertEquals(expected, res);

        v1 = new Vector(Arrays.asList(1.0, 2.0, 3.0), VectorType.COLUMN);
        v2 = new Vector(Arrays.asList(4.0, 5.0, 6.0, 7.0), VectorType.ROW);
        res = v1.mul(v2);
        expected = new Matrix(Arrays.asList(4.0, 5.0, 6.0, 7.0,
                8.0, 10.0, 12.0, 14.0,
                12.0, 15.0, 18.0, 21.0),
                3, 4);

        Assert.assertEquals(expected, res);

        v1 = new Vector(Arrays.asList(1.0, 2.0, 3.0), VectorType.ROW);
        Matrix m1 = new Matrix(Arrays.asList(1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0),
                3, 4);
        Vector result = v1.mul(m1);
        Vector expectedResult = new Vector(Arrays.asList(38.0, 44.0, 50.0, 56.0), VectorType.ROW);

        Assert.assertEquals(expectedResult, result);
    }
}
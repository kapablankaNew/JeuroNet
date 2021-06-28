package kapablankaNew.JeuroNet.Mathematical;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class MatrixTest {

    @Test
    public void addTest() throws VectorMatrixException {
        Matrix m1 = new Matrix(Arrays.asList(1.0, 2.0, 3.0,
                4.0, 5.0, 6.0), 2, 3);
        Matrix m2 = new Matrix(Arrays.asList(-2.0, 1.0, -1.0,
                2.0, -3.0, -1.0), 2, 3);
        Matrix res = m1.add(m2);
        Matrix expected = new Matrix(Arrays.asList(-1.0, 3.0, 2.0,
                6.0, 2.0, 5.0), 2, 3);

        Assert.assertEquals(expected, res);
    }

    @Test
    public void subTest() throws VectorMatrixException {
        Matrix m1 = new Matrix(Arrays.asList(1.0, 2.0, 3.0,
                4.0, 5.0, 6.0), 2, 3);
        Matrix m2 = new Matrix(Arrays.asList(-2.0, 1.0, -1.0,
                2.0, -3.0, -1.0), 2, 3);
        Matrix res = m1.sub(m2);
        Matrix expected = new Matrix(Arrays.asList(3.0, 1.0, 4.0,
                2.0, 8.0, 7.0), 2, 3);

        Assert.assertEquals(expected, res);
    }

    @Test
    public void mulTest() throws VectorMatrixException {
        Matrix m1 = new Matrix(Arrays.asList(1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0), 4, 3);
        Vector v2 = new Vector(Arrays.asList(1.0, 2.0, 3.0), VectorType.COLUMN);
        Vector res = m1.mul(v2);
        Vector expected = new Vector(Arrays.asList(14.0, 32.0, 50.0, 68.0), VectorType.COLUMN);

        Assert.assertEquals(expected, res);

        m1 = new Matrix(Arrays.asList(1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0), 3, 2);
        Matrix m2 = new Matrix(Arrays.asList(1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0), 2, 4);
        Matrix expectedMatrix = new Matrix(Arrays.asList(11.0, 14.0, 17.0, 20.0,
                23.0, 30.0, 37.0, 44.0,
                35.0, 46.0, 57.0, 68.0), 3, 4);
        Matrix result = m1.mul(m2);

        Assert.assertEquals(expectedMatrix, result);

        m1 = new Matrix(Arrays.asList(1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0), 3, 2);
        result = m1.mul(-2.0);
        expectedMatrix = new Matrix(Arrays.asList(-2.0, -4.0,
                -6.0, -8.0,
                -10.0, -12.0), 3, 2);

        Assert.assertEquals(expectedMatrix, result);

        m1 = new Matrix(Arrays.asList(1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0), 3, 2);
        m2 = new Matrix(Arrays.asList(3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0), 3, 2);
        result = m1.mulElemByElem(m2);
        expectedMatrix = new Matrix(Arrays.asList(3.0, 8.0,
                15.0, 24.0,
                35.0, 48.0), 3, 2);

        Assert.assertEquals(expectedMatrix, result);
    }

    @Test
    public void transposeTest() throws VectorMatrixException {
        Matrix m1 = new Matrix(Arrays.asList(1.0, 2.0, 3.0,
                4.0, 5.0, 6.0), 2, 3);
        Matrix result = m1.T();
        Matrix expectedMatrix = new Matrix(Arrays.asList(1.0, 4.0,
                2.0, 5.0,
                3.0, 6.0), 3, 2);

        Assert.assertEquals(expectedMatrix, result);
    }
}
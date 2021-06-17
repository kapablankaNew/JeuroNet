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
    }
}
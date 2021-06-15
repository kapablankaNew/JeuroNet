package kapablankaNew.JeuroNet.Mathematical;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class MatrixTest {

    @Test
    public void add() throws VectorMatrixException {
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
    public void sub() throws VectorMatrixException {
        Matrix m1 = new Matrix(Arrays.asList(1.0, 2.0, 3.0,
                4.0, 5.0, 6.0), 2, 3);
        Matrix m2 = new Matrix(Arrays.asList(-2.0, 1.0, -1.0,
                2.0, -3.0, -1.0), 2, 3);
        Matrix res = m1.sub(m2);
        Matrix expected = new Matrix(Arrays.asList(3.0, 1.0, 4.0,
                2.0, 8.0, 7.0), 2, 3);

        Assert.assertEquals(expected, res);
    }
}
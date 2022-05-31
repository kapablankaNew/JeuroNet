package kapablankaNew.JeuroNet.Mathematical;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@EqualsAndHashCode
public class Matrix implements Serializable {
    @Getter
    private final int rows;

    @Getter
    private final int columns;

    @Getter
    private final double[] elements;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        elements = new double[rows*columns];
    }

    public Matrix (int rows, int columns, @NonNull List<List<Double>> elements) throws VectorMatrixException {
        if (elements.size() != rows) {
            throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
        }
        for (int i = 0; i < rows; i++) {
            if (elements.get(i).size() != columns) {
                throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
            }
        }
        this.rows = rows;
        this.columns = columns;

        this.elements = new double[rows*columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.elements[i*this.columns + j] = elements.get(i).get(j);
            }
        }
    }

    public Matrix(@NonNull List<Double> elements, int rows, int columns) throws VectorMatrixException {
        if (elements.size() != rows * columns) {
            throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
        }
        this.rows = rows;
        this.columns = columns;

        this.elements = new double[rows*columns];
        for (int i = 0; i < rows*columns; i++) {
            this.elements[i] = elements.get(i);
        }
    }

    public Matrix (int rows, int columns, @NonNull double[][] elements) throws VectorMatrixException {
        if (elements.length != rows) {
            throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
        }
        for (int i = 0; i < rows; i++) {
            if (elements[i].length != columns) {
                throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
            }
        }
        this.rows = rows;
        this.columns = columns;

        this.elements = new double[rows*columns];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(elements[i], 0, this.elements, i * columns, columns);
        }
    }

    public Matrix (int rows, int columns, @NonNull double[] elements) throws VectorMatrixException {
        if (elements.length != rows * columns) {
            throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
        }

        this.rows = rows;
        this.columns = columns;

        this.elements = new double[rows*columns];
        System.arraycopy(elements, 0, this.elements, 0, rows * columns);
    }

    public Matrix(@NonNull Matrix other) throws VectorMatrixException {
        this(other.rows, other.columns, other.getElements());
    }

    public Matrix add(@NonNull Matrix matrix) throws VectorMatrixException {
        if (this.rows != matrix.rows || this.columns != matrix.columns) {
            throw new VectorMatrixException("It's not possible to add matrices with different sizes!");
        }
        final int size = this.rows * this.columns;
        double[] result = new double[size];
        double[] first = this.elements;
        double[] second = matrix.elements;
        for (int i = 0; i < size; i++) {
            result[i] = first[i] + second[i];
        }
        return new Matrix(this.rows, this.columns, result);
    }

    public Matrix sub(@NonNull Matrix matrix) throws VectorMatrixException {
        if (this.rows != matrix.rows || this.columns != matrix.columns) {
            throw new VectorMatrixException("It's not possible to add matrices with different sizes!");
        }
        final int size = this.rows * this.columns;
        double[] result = new double[size];
        double[] first = this.elements;
        double[] second = matrix.elements;
        for (int i = 0; i < size; i++) {
            result[i] = first[i] - second[i];
        }
        return new Matrix(this.rows, this.columns, result);
    }

    public Matrix mul(double value) throws VectorMatrixException {
        final int size = this.rows*this.columns;
        double[] result = new double[size];
        double[] elements = this.elements;
        for (int i = 0; i < size; i++) {
            result[i] = elements[i] * value;
        }
        return new Matrix(this.rows, this.columns, result);
    }

    public Vector mul(@NonNull Vector vector) throws VectorMatrixException {
        if (vector.getType() != VectorType.COLUMN) {
            throw new VectorMatrixException("It's not possible to multiply matrix and vector-row!");
        }
        if (this.columns != vector.size()) {
            throw new VectorMatrixException("It's not possible to multiply matrix and vector with different sizes!");
        }
        final int rows = this.rows;
        final int columns = this.columns;
        double[] first = this.elements;
        double[] second = vector.getElements();
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            double elem = 0.0;
            for (int j = 0; j < columns; j++) {
                elem += second[j] * first[i*columns + j];
            }
            result[i] = elem;
        }
        return new Vector(result, VectorType.COLUMN);
    }

    public Matrix mul(@NonNull Matrix matrix) throws VectorMatrixException {
        final int rowsA = this.rows;
        final int columnsA = this.columns;
        final int rowsB = matrix.rows;
        final int columnsB = matrix.columns;

        if (columnsA != rowsB) {
            throw new VectorMatrixException("It's not possible to multiply matrices with different sizes!");
        }
        double[] first = this.elements;
        double[] second = (matrix.T()).elements;
        double[] result = new double[rowsA*columnsB];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < columnsB; j++) {
                double elem = 0.0;
                for (int k = 0; k < columnsA; k++) {
                    elem += first[i*columnsA + k] * second[j*rowsB + k];
                }
                result[i*columnsB + j] = elem;
            }
        }
        return new Matrix(this.rows, matrix.columns, result);
    }

    public Matrix mulElemByElem(@NonNull Matrix matrix) throws VectorMatrixException {
        if (this.rows != matrix.rows || this.columns != matrix.columns) {
            throw new VectorMatrixException("It's not possible to add matrices with different sizes!");
        }
        final int size = this.rows * this.columns;
        double[] result = new double[size];
        double[] first = this.elements;
        double[] second = matrix.elements;
        for (int i = 0; i < size; i++) {
            result[i] = first[i] * second[i];
        }
        return new Matrix(this.rows, this.columns, result);
    }

    public Matrix pow(int n) throws VectorMatrixException {
        if (n == 1) {
            return new Matrix(this);
        }
        else {
            return this.mul(this.pow(n - 1));
        }
    }

    //method for transposing
    public Matrix T() throws VectorMatrixException {
        final int rows = this.rows;
        final int columns = this.columns;
        double[] result = new double[columns*rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0;j < columns; j++) {
                result[j*rows + i] = elements[i*columns + j];
            }
        }
        return new Matrix(this.columns, this.rows, result);
    }

    //Method for limiting values in a vector. For example, if m1 is Matrix [[1, 2, 3], [4, 5, 6]]
    //and res = m1.limit(2, 4), then res is Matrix [[2, 2, 3], [4, 4, 4]]
    public Matrix limit (double min, double max) throws VectorMatrixException {
        final int size = this.rows * this.columns;
        double[] result = new double[size];
        double[] initial = this.elements;
        for (int i = 0; i < size; i++) {
            if (initial[i] > max) {
                result[i] = max;
            } else if (initial[i] < min) {
                result[i] = min;
            } else {
                result[i] = initial[i];
            }
        }
        return new Matrix(this.rows, this.columns, result);
    }

    public double get(int row, int column) {
        return elements[row*columns + column];
    }

    protected double[] getRow(int row) {
        final int columns = this.columns;
        double[] result = new double[columns];
        System.arraycopy(elements, row * columns, result, 0, columns);
        return result;
    }

    protected double[] getColumn(int column) {
        final int rows = this.rows;
        final int columns = this.columns;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = elements[i*columns + column];
        }
        return result;
    }

    public double[][] getElementsAsTwoDimensionalArray() {
        final int rows = this.rows;
        final int columns = this.columns;
        double[] elements = this.elements;
        double[][] result = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(elements, i * columns, result[i], 0, columns);
        }
        return result;
    }
    @Override
    public String toString() {
        double[][] elements = getElementsAsTwoDimensionalArray();
        return "[" + Arrays.stream(elements).map(i ->
                Arrays.stream(i)
                        .mapToObj(Objects::toString)
                        .collect(Collectors.joining(",\t")) + "]").
                collect(Collectors.joining(",\n")) + "]";
    }
}

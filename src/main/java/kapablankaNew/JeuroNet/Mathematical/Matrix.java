package kapablankaNew.JeuroNet.Mathematical;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;
import java.util.ArrayList;
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
    private final double[][] elements;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        elements = new double[rows][columns];
    }

    public Matrix (int rows, int columns, List<List<Double>> elements) throws VectorMatrixException {
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

        this.elements = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.elements[i][j] = elements.get(i).get(j);
            }
        }
    }

    public Matrix(List<Double> elements, int rows, int columns) throws VectorMatrixException {
        if (elements.size() != rows * columns) {
            throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
        }
        this.rows = rows;
        this.columns = columns;

        this.elements = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.elements[i][j] = elements.get(i*columns + j);
            }
        }
    }

    public Matrix (int rows, int columns, double[][] elements) throws VectorMatrixException {
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

        this.elements = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(elements[i], 0, this.elements[i], 0, columns);
        }
    }

    public Matrix(Matrix other) throws VectorMatrixException {
        this(other.getRows(), other.getColumns(), other.getElements());
    }

    public Matrix add(Matrix matrix) throws VectorMatrixException {
        if (this.getRows() != matrix.getRows() || this.getColumns() != matrix.getColumns()) {
            throw new VectorMatrixException("It's not possible to add matrices with different sizes!");
        }
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < this.getRows(); i++) {
            double[] resultRow = result[i];
            double[] firstRow = this.elements[i];
            double[] secondRow = matrix.getElements()[i];
            for (int j = 0; j < this.getColumns(); j++) {
                resultRow[j] = firstRow[j] + secondRow[j];
            }
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
    }

    public Matrix sub(Matrix matrix) throws VectorMatrixException {
        if (this.getRows() != matrix.getRows() || this.getColumns() != matrix.getColumns()) {
            throw new VectorMatrixException("It's not possible to add matrices with different sizes!");
        }
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < this.getRows(); i++) {
            double[] resultRow = result[i];
            double[] firstRow = this.elements[i];
            double[] secondRow = matrix.getElements()[i];
            for (int j = 0; j < this.getColumns(); j++) {
                resultRow[j] = firstRow[j] - secondRow[j];
            }
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
    }

    public Matrix mul(Double value) throws VectorMatrixException {
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < getRows(); i++) {
            double[] row = elements[i];
            double[] resultRow = result[i];
            for (int j = 0; j < getColumns(); j++) {
                resultRow[j] = row[j] * value;
            }
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
    }

    public Vector mul(Vector vector) throws VectorMatrixException {
        if (vector.getType() != VectorType.COLUMN) {
            throw new VectorMatrixException("It's not possible to multiply matrix and vector-row!");
        }
        if (this.getColumns() != vector.size()) {
            throw new VectorMatrixException("It's not possible to multiply matrix and vector with different sizes!");
        }
        double[] result = new double[this.getRows()];

        for (int i = 0; i < this.getRows(); i++) {
            double elem = 0.0;
            double[] firstRow = elements[i];
            for (int j = 0; j < this.getColumns(); j++) {
                elem += vector.get(j) * firstRow[j];
            }
            result[i] = elem;
        }
        List<Double> resultList = new ArrayList<>();
        for (double elem : result) {
            resultList.add(elem);
        }
        return new Vector(resultList, VectorType.COLUMN);
    }

    public Matrix mul(Matrix matrix) throws VectorMatrixException {
        if (this.getColumns() != matrix.getRows()) {
            throw new VectorMatrixException("It's not possible to multiply matrices with different sizes!");
        }
        double[][] result = new double[this.getRows()][matrix.getColumns()];
        for (int i = 0; i < this.getRows(); i++) {
            double[] resultRow = result[i];
            double[] firstRow = this.elements[i];
            for (int j = 0; j < matrix.getColumns(); j++) {
                double[] secondColumn = matrix.getColumn(j);
                double elem = 0.0;
                for (int k = 0; k < this.getColumns(); k++) {
                    elem += firstRow[k] * secondColumn[k];
                }
                resultRow[j] = elem;
            }
        }
        return new Matrix(this.getRows(), matrix.getColumns(), result);
    }

    public Matrix mulElemByElem(Matrix matrix) throws VectorMatrixException {
        if (this.getRows() != matrix.getRows() || this.getColumns() != matrix.getColumns()) {
            throw new VectorMatrixException("It's not possible to add matrices with different sizes!");
        }
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < this.getRows(); i++) {
            double[] resultRow = result[i];
            double[] firstRow = this.elements[i];
            double[] secondRow = matrix.getElements()[i];
            for (int j = 0; j < this.getColumns(); j++) {
                resultRow[j] = firstRow[j] * secondRow[j];
            }
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
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
        double[][] result = new double[getColumns()][getRows()];
        for (int i = 0; i < this.getColumns(); i++) {
            for (int j = 0; j < this.getRows(); j++) {
                result[i][j] = elements[j][i];
            }
        }
        return new Matrix(this.getColumns(), this.getRows(), result);
    }

    //Method for limiting values in a vector. For example, if m1 is Matrix [[1, 2, 3], [4, 5, 6]]
    //and res = m1.limit(2, 4), then res is Matrix [[2, 2, 3], [4, 4, 4]]
    public Matrix limit (double min, double max) throws VectorMatrixException {
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < this.getRows(); i++) {
            double[] row = elements[i];
            double[] resultRow = result[i];
            for (int j = 0; j < this.getColumns(); j++) {
                if (row[j] > max) {
                    resultRow[j] = max;
                } else if (row[j] < min) {
                    resultRow[j] = min;
                } else {
                    resultRow[j] = row[j];
                }
            }
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
    }

    public double get(int row, int column) {
        return elements[row][column];
    }

    protected double[] getRow(int row) {
        return elements[row];
    }

    protected double[] getColumn(int column) {
        double[] result = new double[getRows()];
        for (int i = 0; i < result.length; i++) {
            result[i] = elements[i][column];
        }
        return result;
    }

    @Override
    public String toString() {
        return "[" + Arrays.stream(this.getElements()).map(i ->
                Arrays.stream(i)
                        .mapToObj(Objects::toString)
                        .collect(Collectors.joining(",\t")) + "]").
                collect(Collectors.joining(",\n")) + "]";
    }
}

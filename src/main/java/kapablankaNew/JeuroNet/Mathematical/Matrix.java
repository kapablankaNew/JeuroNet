package kapablankaNew.JeuroNet.Mathematical;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.Serializable;
import java.util.ArrayList;
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
    private final List<List<Double>> elements;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        elements = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                row.add(0.0);
            }
            elements.add(row);
        }
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

        this.elements = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                row.add(elements.get(i).get(j));
            }
            this.elements.add(row);
        }
    }

    public Matrix(List<Double> elements, int rows, int columns) throws VectorMatrixException {
        if (elements.size() != rows * columns) {
            throw new VectorMatrixException("The size of the data is not equal to the size of the matrix.");
        }
        this.rows = rows;
        this.columns = columns;

        this.elements = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                row.add(elements.get(i*columns + j));
            }
            this.elements.add(row);
        }
    }

    public Matrix(Matrix other) throws VectorMatrixException {
        this(other.getRows(), other.getColumns(), other.getElements());
    }

    public Matrix add(Matrix matrix) throws VectorMatrixException {
        if (this.getRows() != matrix.getRows() || this.getColumns() != matrix.getColumns()) {
            throw new VectorMatrixException("It's not possible to add matrices with different sizes!");
        }
        List<List<Double>> result = new ArrayList<>();
        for (int i = 0; i < this.getRows(); i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < this.getColumns(); j++) {
                row.add(this.get(i, j) + matrix.get(i, j));
            }
            result.add(row);
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
    }

    public Matrix sub(Matrix matrix) throws VectorMatrixException {
        if (this.getRows() != matrix.getRows() || this.getColumns() != matrix.getColumns()) {
            throw new VectorMatrixException("It's not possible to subtract matrices with different sizes!");
        }
        List<List<Double>> result = new ArrayList<>();
        for (int i = 0; i < this.getRows(); i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < this.getColumns(); j++) {
                row.add(this.get(i, j) - matrix.get(i, j));
            }
            result.add(row);
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
    }

    public Matrix mul(Double value) throws VectorMatrixException {
        List<List<Double>> result = new ArrayList<>();
        for (List<Double> oldRow : this.elements) {
            List<Double> row = new ArrayList<>();
            for (Double val : oldRow) {
                row.add(val * value);
            }
            result.add(row);
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
        List<Double> result = new ArrayList<>();

        for (int i = 0; i < this.getRows(); i++) {
            double elem = 0.0;
            for (int j = 0; j < this.getColumns(); j++) {
                elem += vector.get(j) * this.get(i, j);
            }
            result.add(elem);
        }

        return new Vector(result, VectorType.COLUMN);
    }

    public Matrix mul(Matrix matrix) throws VectorMatrixException {
        if (this.getColumns() != matrix.getRows()) {
            throw new VectorMatrixException("It's not possible to multiply matrices with different sizes!");
        }
        List<List<Double>> result = new ArrayList<>();
        for (int i = 0; i < this.getRows(); i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < matrix.getColumns(); j++) {
                double elem = 0.0;
                for (int k = 0; k < this.getColumns(); k++) {
                    elem += this.get(i, k) * matrix.get(k, j);
                }
                row.add(elem);
            }
            result.add(row);
        }
        return new Matrix(this.getRows(), matrix.getColumns(), result);
    }

    public Matrix mulElemByElem(Matrix matrix) throws VectorMatrixException {
        if (this.getRows() != matrix.getRows() || this.getColumns() != matrix.getColumns()) {
            throw new VectorMatrixException("It's not possible to multiply element-by-element matrices of different sizes!");
        }
        List<List<Double>> result = new ArrayList<>();

        for (int i = 0; i < this.getRows(); i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < this.getColumns(); j++) {
                row.add(this.get(i, j) * matrix.get(i, j));
            }
            result.add(row);
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
        List<List<Double>> result = new ArrayList<>();
        for (int i = 0; i < this.getColumns(); i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < this.getRows(); j++) {
                row.add(this.get(j, i));
            }
            result.add(row);
        }
        return new Matrix(this.getColumns(), this.getRows(), result);
    }

    //Method for limiting values in a vector. For example, if m1 is Matrix [[1, 2, 3], [4, 5, 6]]
    //and res = m1.limit(2, 4), then res is Matrix [[2, 2, 3], [4, 4, 4]]
    public Matrix limit (double min, double max) throws VectorMatrixException {
        List<List<Double>> result = new ArrayList<>();
        for (int i = 0; i < this.getRows(); i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < this.getColumns(); j++) {
                if (this.get(i, j) > max) {
                    row.add(max);
                } else if (this.get(i, j) < min) {
                    row.add(min);
                } else {
                    row.add(this.get(i, j));
                }
            }
            result.add(row);
        }
        return new Matrix(this.getRows(), this.getColumns(), result);
    }

    public double get(int row, int column) {
        return elements.get(row).get(column);
    }

    @Override
    public String toString() {
        return "[" + this.getElements().
                stream().
                map(s -> "[" + s.stream().
                        map(Objects::toString).
                        collect(Collectors.joining(",\t")) + "]").
                collect(Collectors.joining(",\n")) + "]";
    }
}

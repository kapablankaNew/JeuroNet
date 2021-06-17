package kapablankaNew.JeuroNet.Mathematical;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode
public class Matrix {
    @Getter
    private final int rows;

    @Getter
    private final int columns;

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

    public Vector mul(Vector vector) throws VectorMatrixException{
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

    public double get(int row, int column) {
        return elements.get(row).get(column);
    }
}

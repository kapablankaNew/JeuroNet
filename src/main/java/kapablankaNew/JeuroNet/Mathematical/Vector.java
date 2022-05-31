package kapablankaNew.JeuroNet.Mathematical;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

@EqualsAndHashCode
public class Vector implements Serializable {
    @Getter
    private final double[] elements;

    @Getter
    private final VectorType type;

    public Vector(int size) {
        this(size, VectorType.COLUMN);
    }

    public Vector(int size, @NonNull VectorType type) {
        elements = new double[size];
        this.type = type;
    }

    public Vector(@NonNull List<Double> elements) {
        this(elements, VectorType.COLUMN);
    }

    public Vector(@NonNull List<Double> elements, @NonNull VectorType type) {
        final int size = elements.size();
        this.elements = new double[size];
        for(int i = 0; i < size; i++) {
            this.elements[i] = elements.get(i);
        }
        this.type = type;
    }

    public Vector(@NonNull double[] elements) {
        this(elements, VectorType.COLUMN);
    }

    public Vector(@NonNull double[] elements, @NonNull VectorType type) {
        this.elements = new double[elements.length];
        System.arraycopy(elements, 0, this.elements, 0, elements.length);
        this.type = type;
    }


    public Vector(@NonNull Vector vector) {
        this(vector, vector.getType());
    }

    public Vector(@NonNull Vector vector, @NonNull VectorType type) {
        this.elements = new double[vector.size()];
        double[] elements = vector.elements;
        System.arraycopy(elements, 0, this.elements, 0, this.elements.length);
        this.type = type;
    }

    public static Vector getVectorWithElementsOfOne(int size, @NonNull VectorType type) {
        double[] elements = new double[size];
        for (int i = 0; i < size; i++) {
            elements[i] = 1.0;
        }
        return new Vector(elements, type);
    }

    public int size() {
        return elements.length;
    }

    public double get(int index) {
        return elements[index];
    }

    public List<Double> getElementsAsList() {
        return DoubleStream.of(elements).boxed().collect(Collectors.toCollection(ArrayList::new));
    }
    public Vector add(@NonNull Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        final int size = this.size();
        if (size != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = this.elements[i] + vector.elements[i];
        }
        return new Vector(result, this.getType());
    }

    public Vector sub(@NonNull Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        final int size = this.size();
        if (size != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = this.elements[i] - vector.elements[i];
        }
        return new Vector(result, this.getType());
    }

    public Vector mul(double value) {
        final int size = this.size();
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = this.elements[i] * value;
        }
        return new Vector(result, this.getType());
    }

    public Matrix mul(@NonNull Vector vector) throws VectorMatrixException {
        final int rows = this.size();
        final int columns = vector.size();
        double[] result = new double[rows*columns];
        if (this.getType() == vector.getType()) {
            throw new VectorMatrixException("It's not possible to multiply two vector-rows or two vector-columns!");
        }
        double[] firstVector = this.elements;
        double[] secondVector = vector.elements;
        if (this.getType() == VectorType.ROW && vector.getType() == VectorType.COLUMN) {
            if (rows != columns) {
                throw new VectorMatrixException("It's not possible to multiply vectors of different sizes!");
            }
            double elem = 0.0;
            for (int i = 0; i < rows; i++) {
                elem += firstVector[i]*secondVector[i];
            }
            return new Matrix(Collections.singletonList(elem), 1, 1);
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i*columns + j] = firstVector[i] * secondVector[j];
            }
        }
        return new Matrix(rows, columns, result);
    }

    public Vector mul(@NonNull Matrix matrix) throws VectorMatrixException {
        if (this.getType() != VectorType.ROW) {
            throw new VectorMatrixException("It's not possible to multiply vector-column and matrix!");
        }
        final int size = this.size();
        final int rows = matrix.getRows();
        final int columns = matrix.getColumns();
        if (size != rows) {
            throw new VectorMatrixException("It's not possible to multiply vector and matrix with different sizes!");
        }
        double[] result = new double[columns];

        for (int i = 0; i < columns; i++) {
            double[] row = this.elements;
            double[] column = matrix.getColumn(i);
            double elem = 0.0;
            for (int j = 0; j < rows; j++) {
                elem += row[j]*column[j];
            }
            result[i] = elem;
        }
        return new Vector(result, VectorType.ROW);
    }

    /*
    This method using for multiplying vectors element-by-element
    For example, if v1 is Vector [1, 2, 3], and v2 is Vector [4, 5, 6], then
    result = v1.mulElemByElem(v2) is Vector [4, 10, 18]
     */
    public Vector mulElemByElem (@NonNull Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to multiply element-by-element vector-row and vector-column! ");
        }
        final int size = this.size();
        if (size != vector.size()) {
            throw new VectorMatrixException("It's not possible to multiply element-by-element vectors of different sizes!");
        }
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = this.elements[i] * vector.elements[i];
        }
        return new Vector(result, this.getType());
    }

    //method for transposing
    public Vector T() {
        VectorType result;
        if (this.getType() == VectorType.ROW) {
            result = VectorType.COLUMN;
        } else {
            result = VectorType.ROW;
        }
        return new Vector(this, result);
    }

    //Method for limiting values in a vector. For example, if v1 is Vector [1, 2, 3, 4, 5]
    //and res = v1.limit(2, 4), then res is Vector [2, 2, 3, 4, 4]
    public Vector limit(double min, double max) {
        final int size = this.size();
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            if (this.get(i) > max) {
                result[i] = max;
            } else if (this.get(i) < min) {
                result[i] = min;
            } else {
                result[i] = this.get(i);
            }
        }
        return new Vector(result, this.getType());
    }

    @Override
    public String toString() {
        return "[" +
                Arrays.stream(elements).
                        mapToObj(Objects::toString).
                        collect(Collectors.joining("; "))
                + "]";
    }
}

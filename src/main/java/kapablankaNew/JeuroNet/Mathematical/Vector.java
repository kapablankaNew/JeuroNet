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
import java.util.stream.IntStream;

@EqualsAndHashCode
public class Vector implements Serializable {
    @Getter
    private final double[] elements;

    @Getter
    private final VectorType type;

    public Vector(int size) {
        this(size, VectorType.COLUMN);
    }

    public Vector(int size, VectorType type) {
        elements = new double[size];
        this.type = type;
    }

    public Vector(List<Double> elements) {
        this(elements, VectorType.COLUMN);
    }

    public Vector(List<Double> elements, VectorType type) {
        this.elements = new double[elements.size()];
        for(int i = 0; i < elements.size(); i++) {
            this.elements[i] = elements.get(i);
        }
        this.type = type;
    }

    public Vector(double[] elements) {
        this(elements, VectorType.COLUMN);
    }

    public Vector(double[] elements, VectorType type) {
        this.elements = new double[elements.length];
        System.arraycopy(elements, 0, this.elements, 0, elements.length);
        this.type = type;
    }


    public Vector(@NonNull Vector vector) {
        this(vector, vector.getType());
    }

    public Vector(@NonNull Vector vector, VectorType type) {
        this.elements = new double[vector.size()];
        double[] elements = vector.elements;
        System.arraycopy(elements, 0, this.elements, 0, this.elements.length);
        this.type = type;
    }

    public static Vector getVectorWithElementsOfOne(int size, VectorType type) {
        List<Double> elements = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            elements.add(1.0);
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
    public Vector add(Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        if (this.size() != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        double[] result = new double[this.size()];
        for (int i = 0; i < this.size(); i++) {
            result[i] = this.elements[i] + vector.elements[i];
        }
        return new Vector(result, this.getType());
    }

    public Vector sub(Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        if (this.size() != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        double[] result = new double[this.size()];
        for (int i = 0; i < this.size(); i++) {
            result[i] = this.elements[i] - vector.elements[i];
        }
        return new Vector(result, this.getType());
    }

    public Vector mul(double value) {
        double[] result = new double[this.size()];
        for (int i = 0; i < this.size(); i++) {
            result[i] = this.elements[i] * value;
        }
        return new Vector(result, this.getType());
    }

    public Matrix mul(Vector vector) throws VectorMatrixException {
        double[][] result = new double[this.size()][vector.size()];
        if (this.getType() == vector.getType()) {
            throw new VectorMatrixException("It's not possible to multiply two vector-rows or two vector-columns!");
        }
        double[] firstVector = this.elements;
        double[] secondVector = vector.elements;
        if (this.getType() == VectorType.ROW && vector.getType() == VectorType.COLUMN) {
            if (this.size() != vector.size()) {
                throw new VectorMatrixException("It's not possible to multiply vectors of different sizes!");
            }
            double elem = 0.0;
            for (int i = 0; i < this.size(); i++) {
                elem += firstVector[i]*secondVector[i];
            }
            return new Matrix(Collections.singletonList(elem), 1, 1);
        }

        for (int i = 0; i < this.size(); i++) {
            double[] resultRow = result[i];
            for (int j = 0; j < vector.size(); j++) {
                resultRow[j] = firstVector[i] * secondVector[j];
            }
        }
        return new Matrix(this.size(), vector.size(), result);
    }

    public Vector mul(Matrix matrix) throws VectorMatrixException {
        if (this.getType() != VectorType.ROW) {
            throw new VectorMatrixException("It's not possible to multiply vector-column and matrix!");
        }
        if (this.size() != matrix.getRows()) {
            throw new VectorMatrixException("It's not possible to multiply vector and matrix with different sizes!");
        }
        double[] result = new double[matrix.getColumns()];

        for (int i = 0; i < result.length; i++) {
            double[] row = this.elements;
            double[] column = matrix.getColumn(i);
            double elem = 0.0;
            for (int j = 0; j < matrix.getRows(); j++) {
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
    public Vector mulElemByElem (Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to multiply element-by-element vector-row and vector-column! ");
        }
        if (this.size() != vector.size()) {
            throw new VectorMatrixException("It's not possible to multiply element-by-element vectors of different sizes!");
        }
        double[] result = new double[this.size()];
        for (int i = 0; i < this.size(); i++) {
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
        double[] result = new double[this.size()];
        for (int i = 0; i < this.size(); i++) {
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

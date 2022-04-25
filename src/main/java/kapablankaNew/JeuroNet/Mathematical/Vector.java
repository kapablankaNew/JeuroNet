package kapablankaNew.JeuroNet.Mathematical;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@EqualsAndHashCode
public class Vector implements Serializable {
    @Getter
    private final List<Double> elements;

    @Getter
    private final VectorType type;

    public Vector(int size) {
        this(size, VectorType.COLUMN);
    }

    public Vector(int size, VectorType type) {
        elements = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            elements.add(0.0);
        }
        this.type = type;
    }

    public Vector(List<Double> elements) {
        this(elements, VectorType.COLUMN);
    }

    public Vector(List<Double> elements, VectorType type) {
        this.elements = new ArrayList<>();
        this.elements.addAll(elements);
        this.type = type;
    }

    public Vector(@NonNull Vector vector) {
        this(vector, vector.getType());
    }

    public Vector(@NonNull Vector vector, VectorType type) {
        this.elements = new ArrayList<>();
        this.elements.addAll(vector.elements);
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
        return elements.size();
    }

    public double get(int index) {
        return elements.get(index);
    }

    public Vector add(Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        if (this.size() != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        List<Double> result = IntStream.range(0, size()).
                mapToObj(i -> get(i) + vector.get(i)).
                collect(Collectors.toList());

        return new Vector(result, this.getType());
    }

    public Vector sub(Vector vector) throws VectorMatrixException {
        if (this.getType() != vector.getType()) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        if (this.size() != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        List<Double> result = IntStream.range(0, size()).
                mapToObj(i -> get(i) - vector.get(i)).
                collect(Collectors.toList());

        return new Vector(result, this.getType());
    }

    public Vector mul(double value) {
        List<Double> result = new ArrayList<>();
        for (Double elem : this.getElements()) {
            result.add(elem * value);
        }
        return new Vector(result, this.getType());
    }

    public Matrix mul(Vector vector) throws VectorMatrixException {
        List<List<Double>> result = new ArrayList<>();
        if (this.getType() == vector.getType()) {
            throw new VectorMatrixException("It's not possible to multiply two vector-rows or two vector-columns!");
        }
        if (this.getType() == VectorType.ROW && vector.getType() == VectorType.COLUMN) {
            if (this.size() != vector.size()) {
                throw new VectorMatrixException("It's not possible to multiply vectors of different sizes!");
            }
            double elem = IntStream.range(0, this.size()).
                    mapToDouble(i -> this.get(i) * vector.get(i)).
                    sum();
            result.add(Collections.singletonList(elem));
            return new Matrix(1, 1, result);
        }

        for (int i = 0; i < this.size(); i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < vector.size(); j++) {
                row.add(this.get(i) * vector.get(j));
            }
            result.add(row);
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
        List<Double> result = new ArrayList<>();

        for (int i = 0; i < matrix.getColumns(); i++) {
            double elem = 0.0;
            for (int j = 0; j < matrix.getRows(); j++) {
                elem += this.get(j) * matrix.get(j, i);
            }
            result.add(elem);
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
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < this.size(); i++) {
            result.add(this.get(i) * vector.get(i));
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
        List<Double> result = new ArrayList<>();
        for (int i = 0; i < this.size(); i++) {
            if (this.get(i) > max) {
                result.add(max);
            } else if (this.get(i) < min) {
                result.add(min);
            } else {
                result.add(this.get(i));
            }
        }
        return new Vector(result, this.getType());
    }

    @Override
    public String toString() {
        return "[" +
                elements.stream().
                        map(Objects::toString).
                        collect(Collectors.joining("; "))
                + "]";
    }
}

package kapablankaNew.JeuroNet.Mathematical;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@EqualsAndHashCode
public class Vector {
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

    public Vector(Vector vector) {
        this(vector, vector.type);
    }

    public Vector(Vector vector, VectorType type) {
        elements = new ArrayList<>();
        elements.addAll(vector.elements);
        this.type = type;
    }

    public int size() {
        return elements.size();
    }

    public double get(int index) {
        return elements.get(index);
    }

    public Vector add(Vector vector) throws VectorMatrixException {
        if (this.type != vector.type) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        if (this.size() != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        List<Double> result = IntStream.range(0, size()).
                mapToObj(i -> get(i) + vector.get(i)).
                collect(Collectors.toList());

        return new Vector(result, this.type);
    }

    public Vector sub(Vector vector) throws VectorMatrixException {
        if (this.type != vector.type) {
            throw new VectorMatrixException("It's not possible to add vector-row and vector-column!");
        }
        if (this.size() != vector.size()) {
            throw new VectorMatrixException("It's not possible to add vectors of different sizes!");
        }

        List<Double> result = IntStream.range(0, size()).
                mapToObj(i -> get(i) - vector.get(i)).
                collect(Collectors.toList());

        return new Vector(result, this.type);
    }
}

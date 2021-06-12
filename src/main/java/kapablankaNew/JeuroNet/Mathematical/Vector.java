package kapablankaNew.JeuroNet.Mathematical;

import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

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
}

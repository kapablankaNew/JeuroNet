package kapablankaNew.JeuroNet;

import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TextConverterTest {

    @Test
    public void convertTest() {
        TextConverter converter = new TextConverter(Arrays.asList("I am good",
                "I am", "I"));
        List<Vector> result = converter.convert("Am good I");
        List<Vector> expected = new ArrayList<>();
        Vector exp = new Vector(Arrays.asList(0.0, 1.0, 0.0), VectorType.COLUMN);
        expected.add(exp);
        exp = new Vector(Arrays.asList(0.0, 0.0, 1.0), VectorType.COLUMN);
        expected.add(exp);
        exp = new Vector(Arrays.asList(1.0, 0.0, 0.0), VectorType.COLUMN);
        expected.add(exp);
        assertEquals(expected, result);
    }
}
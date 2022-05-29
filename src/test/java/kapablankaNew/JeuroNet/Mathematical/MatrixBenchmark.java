package kapablankaNew.JeuroNet.Mathematical;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.junit.jupiter.api.Test;
import org.openjdk.jmh.Main;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
@Fork(value = 3, warmups = 3, jvmArgs = {"-Xms2G", "-Xmx2G"})
public class MatrixBenchmark {
    @Param({"10", "100", "500", "1000"})
    public int size;

    Matrix a;
    Matrix b;

    @Test
    public void testT() throws VectorMatrixException, ExecutionException, InterruptedException {
        size = 100;
        for (int i = 0; i < 10; i++) {
            setUp();
            Instant start = Instant.now();
            //var start = System.nanoTime();
            Matrix result = a.mul(b);
            //var finish = System.nanoTime();
            Instant finish = Instant.now();
            var n = result.getRows();
            System.out.println(n);
            Duration elapsed = Duration.between(start, finish);
            String format = "ss.SSS";
            System.out.println("Время, мс: " + DurationFormatUtils.formatDuration(elapsed.toMillis(), format));
            //long duration = finish - start;
            //double milli = duration/1000.0;
            //System.out.println("Time, millisec: " + milli);
        }
    }

    @Setup(Level.Invocation)
    public void setUp() {
        try {
            a = getRandomSquareMatrix(size);
            b = getRandomSquareMatrix(size);
        } catch (VectorMatrixException e) {
            a = new Matrix(size, size);
            b = new Matrix(size, size);
            System.out.println("Matrix error!");
        }
    }

    public static void main(String[] args) throws IOException {
        Main.main(args);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void testMultiply(Blackhole blackhole) throws VectorMatrixException {
        Matrix res = a.mul(b);
        blackhole.consume(res);
    }

    @Test
    public void test() throws VectorMatrixException, ExecutionException, InterruptedException {
        size = 5000;
        a = getRandomSquareMatrix(size);
        b = getRandomSquareMatrix(size);
        Instant start = Instant.now();
        Matrix res = a.mulElemByElem(b);
        Instant finish = Instant.now();
        Duration elapsed = Duration.between(start, finish);
        String format = "mm:ss.SSS";
        System.out.println("Result time: " + DurationFormatUtils.formatDuration(elapsed.toMillis(), format));
        System.out.println(res.getRows());
    }

    private Matrix getRandomSquareMatrix(int size) throws VectorMatrixException {
        List<List<Double>> elements = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < size; j++) {
                row.add(ThreadLocalRandom.current().nextDouble(-100.0, 100.0));
            }
            elements.add(row);
        }
        return new Matrix(size, size, elements);
    }
}

package kapablankaNew.JeuroNet.MLP;

public class MultiLayerPerceptronException extends Exception {
    public MultiLayerPerceptronException(String Message) {
        super(Message);
    }

    @Override
    public String toString() {
        String result = super.toString();
        return "Error in neural network: " + result;
    }
}

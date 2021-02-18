package kapablankaNew.JeuroNet.MLP;

public class TopologyException extends Exception {
    public TopologyException(String Message) {
        super(Message);
    }

    @Override
    public String toString() {
        String result = super.toString();
        return "Error in neural network: " + result;
    }
}
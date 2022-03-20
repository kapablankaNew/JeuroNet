package kapablankaNew.JeuroNet;

import java.io.IOException;
import java.io.Serializable;

public interface Storable extends Serializable {
    void save(String path) throws IOException;
}

package kapablankaNew.JeuroNet;

import java.util.*;
import kapablankaNew.JeuroNet.Mathematical.Vector;
import kapablankaNew.JeuroNet.Mathematical.VectorType;

/*
This class convert string data to numbers
The conversion uses a unitary code, that is, each word is converted to a vector with
one element 1 and the other 0. The size of the vectors is equal to the number of unique words
in the data. The case is not taken into account.
*/
public class TextConverter {
    private final Map<String, Integer> wordsFrequency;

    public TextConverter(List<String> data) {
        wordsFrequency = new LinkedHashMap<>();
        //find all unique words and count the number of times each word occurs in the data.
        data.stream().map(s -> s.toLowerCase().split(" ")).
                flatMap(Arrays::stream).
                forEach(s -> {
                    if (wordsFrequency.containsKey(s)) {
                        wordsFrequency.put(s, wordsFrequency.get(s) + 1);
                    } else {
                        wordsFrequency.put(s, 1);
                    }
                });
        Map<String, Integer> additional = sortByValue(wordsFrequency);
        wordsFrequency.clear();
        int i = 0;
        //Changing the number of occurrences to the number in the list
        //So, The most frequent word corresponds to 0, the next most frequent word corresponds
        // to 1, and so on.
        for (String key : additional.keySet()) {
            wordsFrequency.put(key, i);
            i++;
        }
    }

    private static Map<String, Integer> sortByValue (Map<String, Integer> unsortedMap) {
        Map<String, Integer> result = new LinkedHashMap<>();
        List<Map.Entry<String, Integer>> list = new ArrayList<>(unsortedMap.entrySet());

        //sort data in reverse order
        list.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));

        for (Map.Entry<String, Integer> entry : list)  {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }

    public List<Vector> convert(String data) {
        List<Vector> result = new ArrayList<>();
        for (String word : data.toLowerCase().split(" ")) {
            List<Double> vector = new ArrayList<>();
            for (int i = 0; i < wordsFrequency.size(); i++) {
                vector.add(0.0);
            }
            vector.set(wordsFrequency.get(word), 1.0);
            result.add(new Vector(vector, VectorType.COLUMN));
        }
        return result;
    }

    public int getNumberUniqueWords() {
        return wordsFrequency.size();
    }
}

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;

public class ArffToCsv {
    public static void main(String[] args) throws Exception {
        ArffLoader arffLoader = new ArffLoader();
        arffLoader.setSource(new File("./code/data/HepatitisCdata.arff"));
        Instances dataset = arffLoader.getDataSet();

        CSVSaver csvSaver = new CSVSaver();
        csvSaver.setInstances(dataset);
        csvSaver.setFile(new File("./code/data/HepatitisCdata1.csv"));
        csvSaver.writeBatch();
    }
}

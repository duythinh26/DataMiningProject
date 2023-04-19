import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CsvToArff {
    public static void main(String[] args) throws Exception {
        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setSource(new File("data/HepatitisCdata.csv"));
        Instances dataset = csvLoader.getDataSet();

        ArffSaver arffSaver = new ArffSaver();
        arffSaver.setInstances(dataset);
        arffSaver.setFile(new File("data/HepatitisCdata.arff"));
        arffSaver.writeBatch();
    }
}

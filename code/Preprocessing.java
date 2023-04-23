import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class Preprocessing {
    public static void main(String[] args) throws Exception{
        // Handle missing values
        DataSource source = new DataSource("./code/data/HepatitisCdata.arff");
        Instances dataset = source.getDataSet();
        ArffSaver saver = new ArffSaver();

        ReplaceMissingValues missingValues = new ReplaceMissingValues();
        missingValues.setInputFormat(dataset);
        Instances noMissing = Filter.useFilter(dataset, missingValues);

        saver.setInstances(noMissing);
        saver.setFile(new File("./code/data/noMissing-HepatitisCdata.arff"));
        saver.writeBatch();

        
    }
}

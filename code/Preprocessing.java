import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class Preprocessing {
    public static void main(String[] args) throws Exception{
        /**
         * Handle missing values
         */
        // Load raw dataset
        DataSource src = new DataSource("./code/data/HepatitisCdata.arff");
        Instances dataset = src.getDataSet();
        ArffSaver saver = new ArffSaver();

        // Create object to handle missing values
        ReplaceMissingValues missingValues = new ReplaceMissingValues();

        // Put the dataset into the filter and use filter
        missingValues.setInputFormat(dataset);
        Instances noMissing = Filter.useFilter(dataset, missingValues);

        // Write a new dataset after finish handling missing values
        saver.setInstances(noMissing);
        saver.setFile(new File("./code/data/noMissing-HepatitisCdata.arff"));
        saver.writeBatch();

        /**
         * Detect outliers and extreme values
         */
        // Load no missing values dataset
        DataSource src1 = new DataSource("./code/data/noMissing-HepatitisCdata.arff");
        Instances dataset1 = src1.getDataSet();

        // Set up the options for interquartile range
        String[] option = new String[]{"-R", "first-last", "-O", "3.0", "-E", "6.0"};

        // Create an object to define interquartile range values
        InterquartileRange interquartileRange = new InterquartileRange();

        // Set the options for filter
        interquartileRange.setOptions(option);

        // Put the dataset into the filter and use filter
        interquartileRange.setInputFormat(dataset1);
        Instances iqrData = Filter.useFilter(dataset1, interquartileRange);

        // Write a new dataset after detect interquartile range values
        saver.setInstances(iqrData);
        saver.setFile(new File("./code/data/IQR-HepatitisCdata.arff"));
        saver.writeBatch();
    }
}

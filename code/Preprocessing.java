import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.RemoveWithValues;

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
        DataSource iqrSrc = new DataSource("./code/data/noMissing-HepatitisCdata.arff");
        Instances iqrDataset = iqrSrc.getDataSet();

        // Set up the options for interquartile range
        String[] option = new String[]{"-R", "first-last", "-O", "1.5", "-E", "3.0"};

        // Create an object to define interquartile range values
        InterquartileRange interquartileRange = new InterquartileRange();

        // Set the options for filter
        interquartileRange.setOptions(option);

        // Put the dataset into the filter and use filter
        interquartileRange.setInputFormat(iqrDataset);
        Instances iqrData = Filter.useFilter(iqrDataset, interquartileRange);

        // Write a new dataset after detect interquartile range values
        saver.setInstances(iqrData);
        saver.setFile(new File("./code/data/IQR-HepatitisCdata.arff"));
        saver.writeBatch();

        /**
         * Remove the outliers and extreme values
         */
        // Load InterQuartile Range values dataset
        DataSource outlierSrc = new DataSource("./code/data/IQR-HepatitisCdata.arff");
        Instances outlierDataset = outlierSrc.getDataSet();

        // Set up the options for remove outlier values
        String[] outlierOption = new String[]{"-S", "0.0", "-C", "15", "-L", "last"};

        // Create an object to remove outlier values
        RemoveWithValues removeOutlier = new RemoveWithValues();

        // Set the options for filter
        removeOutlier.setOptions(outlierOption);

        // Put the dataset into the filter and use filter
        removeOutlier.setInputFormat(outlierDataset);
        Instances removeOutlierData = Filter.useFilter(outlierDataset, removeOutlier);

        // Write a new dataset after remove outlier values
        saver.setInstances(removeOutlierData);
        saver.setFile(new File("./code/data/outlierRemoved-HepatitisCdata.arff"));
        saver.writeBatch();

        // Load outlier removed dataset
        DataSource extremeSrc = new DataSource("./code/data/outlierRemoved-HepatitisCdata.arff");
        Instances extremeDataset = extremeSrc.getDataSet();

        // Set up the options for remove outlier values
        String[] extremeOption = new String[]{"-S", "0.0", "-C", "16", "-L", "last"};

        // Create an object to remove outlier values
        RemoveWithValues removeExtreme = new RemoveWithValues();

        // Set the options for filter
        removeExtreme.setOptions(extremeOption);

        // Put the dataset into the filter and use filter
        removeExtreme.setInputFormat(extremeDataset);
        Instances removeExtremeData = Filter.useFilter(extremeDataset, removeExtreme);

        // Write a new dataset after remove outlier values
        saver.setInstances(removeExtremeData);
        saver.setFile(new File("./code/data/extremeRemoved-HepatitisCdata.arff"));
        saver.writeBatch();
    }
}

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
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

        /*
         * Convert numeric to nominal
         */
        // Load raw dataset
        src = new DataSource("./code/data/noMissing-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Set up the options to convert from numeric to nominal
        String[] num2NomOption = new String[]{"-R", "first-last"};

        // Create object to convert numeric to nominal
        NumericToNominal numericToNominal = new NumericToNominal();

        // Set the filter option
        numericToNominal.setOptions(num2NomOption);

        // Put the dataset into the filter and use filter
        numericToNominal.setInputFormat(dataset);
        Instances numeric2Nominal = Filter.useFilter(dataset, numericToNominal);

        // Write a new dataset after finish handling missing values
        saver.setInstances(numeric2Nominal);
        saver.setFile(new File("./code/data/numericToNominal-HepatitisCdata.arff"));
        saver.writeBatch();

        /**
         * Detect outliers and extreme values
         */
        // Load no missing values dataset
        DataSource iqrSrc = new DataSource("./code/data/numericToNominal-HepatitisCdata.arff");
        Instances iqrDataset = iqrSrc.getDataSet();

        // Set up the options to interquartile range
        String[] option = new String[]{"-R", "first-last", "-O", "1.5", "-E", "3.0"};

        // Create an object to define interquartile range values
        InterquartileRange interquartileRange = new InterquartileRange();

        // Set the filter option
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

        // Set up the options to remove outlier values
        String[] outlierOption = new String[]{"-S", "0.0", "-C", "15", "-L", "last"};

        // Create an object to remove outlier values
        RemoveWithValues removeOutlier = new RemoveWithValues();

        // Set the filter option
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

        // Set up the options to remove extreme values
        String[] extremeOption = new String[]{"-S", "0.0", "-C", "16", "-L", "last"};

        // Create an object to remove extreme values
        RemoveWithValues removeExtreme = new RemoveWithValues();

        // Set the filter option
        removeExtreme.setOptions(extremeOption);

        // Put the dataset into the filter and use filter
        removeExtreme.setInputFormat(extremeDataset);
        Instances removeExtremeData = Filter.useFilter(extremeDataset, removeExtreme);

        // Write a new dataset after remove extreme values
        saver.setInstances(removeExtremeData);
        saver.setFile(new File("./code/data/extremeRemoved-HepatitisCdata.arff"));
        saver.writeBatch();

        /**
         * Remove Outlier and ExtremeValue attributes
         */
        src = new DataSource("./code/data/extremeRemoved-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Set up the options to remove attribute
        String[] opt = new String[]{"-R", "15,16"};

        // Create an object to remove attribute
        Remove remove = new Remove();

        // Set the filter option
        remove.setOptions(opt);

        // Put the dataset into the filter and use filter
        remove.setInputFormat(dataset);
        Instances newData = Filter.useFilter(dataset, remove);

        // Write a new dataset after remove attributes
        saver.setInstances(newData);
        saver.setFile(new File("./code/data/cleaned-HepatitisCdata.arff"));
        saver.writeBatch();

        /*
         * Discretize Attributes
         */
        // Load dataset
        DataSource discretizeSrc = new DataSource("./code/data/cleaned-HepatitisCdata.arff");
        Instances discretizeDataset = discretizeSrc.getDataSet();

        // Set up options to findNumBins, 10 bins, -1.0 desiredWeightOfInstancesPerInterval, 6 binRangePrecision
        String[] discretizeOption = new String[]{"-O", "-B", "10", "-M", "-1.0", "-R", "first-last", "-precision", "6"};

        // Create Discretize object
        Discretize discretize = new Discretize();

        // Set the filter option
        discretize.setOptions(discretizeOption);

        // Put the dataset into the filter and use filter
        discretize.setInputFormat(discretizeDataset);
        newData = Filter.useFilter(discretizeDataset, discretize);

        // Write a new dataset after discretize
        saver.setInstances(newData);
        saver.setFile(new File("./code/data/discretized-HepatitisCdata.arff"));
        saver.writeBatch();
    }
}

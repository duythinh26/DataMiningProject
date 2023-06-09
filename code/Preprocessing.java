import java.io.File;

import weka.attributeSelection.CorrelationAttributeEval;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class Preprocessing {
    public static void main(String[] args) throws Exception{
        /*
         * Convert string to nominal
         */
        // Load raw dataset
        DataSource src = new DataSource("./code/data/HepatitisCdata.arff");
        Instances dataset = src.getDataSet();
        ArffSaver saver = new ArffSaver();

        // Handle "NA" values (change to ? values to easily handle missing)
        int[] replacesAttrs = new int[] { 4, 5, 6, 10, 13 };
        for (int i = 0; i < dataset.numInstances(); i++) {
            for (int j = 0; j < replacesAttrs.length; j++) {
                if (dataset.instance(i).toString(replacesAttrs[j]).equals("NA")) {
                    dataset.instance(i).setMissing(replacesAttrs[j]);
                    ;
                }
            }
        }

        // Create object to convert string to nominal
        StringToNominal StringToNominal = new StringToNominal();

        // Set the filter option
        StringToNominal.setOptions(new String[]{"-R", "first-last"});

        // Put the dataset into the filter and use filter
        StringToNominal.setInputFormat(dataset);
        Instances string2Nominal = Filter.useFilter(dataset, StringToNominal);

        // Write a new dataset after finish handling missing values
        saver.setInstances(string2Nominal);
        saver.setFile(new File("./code/data/StringToNominal-HepatitisCdata.arff"));
        saver.writeBatch();

        /*
         * Handle missing values
         */
        // Load dataset
        src = new DataSource("./code/data/StringToNominal-HepatitisCdata.arff");
        dataset = src.getDataSet();

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
         * Detect outliers and extreme values
         */
        // Load no missing values dataset
        src = new DataSource("./code/data/noMissing-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Create an object to define interquartile range values
        InterquartileRange range = new InterquartileRange();

        // Set the filter option
        range.setOptions(new String[]{"-R", "first-last", "-O", "1.5", "-E", "3.0"});

        // Put the dataset into the filter and use filter
        range.setInputFormat(dataset);
        Instances iqrData = Filter.useFilter(dataset, range);

        // Write a new dataset after detect interquartile range values
        saver.setInstances(iqrData);
        saver.setFile(new File("./code/data/IQR-HepatitisCdata.arff"));
        saver.writeBatch();

        /*
         * Remove the outliers and extreme values
         */
        // Load InterQuartile Range values dataset
        src = new DataSource("./code/data/IQR-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Create an object to remove outlier values
        RemoveWithValues removeOutlier = new RemoveWithValues();

        // Set the filter option
        removeOutlier.setOptions(new String[]{"-S", "0.0", "-C", "15", "-L", "last"});

        // Put the dataset into the filter and use filter
        removeOutlier.setInputFormat(dataset);
        Instances removeOutlierData = Filter.useFilter(dataset, removeOutlier);

        // Write a new dataset after remove outlier values
        saver.setInstances(removeOutlierData);
        saver.setFile(new File("./code/data/outlierRemoved-HepatitisCdata.arff"));
        saver.writeBatch();

        // Load outlier removed dataset
        src = new DataSource("./code/data/outlierRemoved-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Create an object to remove extreme values
        RemoveWithValues removeExtreme = new RemoveWithValues();

        // Set the filter option
        removeExtreme.setOptions(new String[]{"-S", "0.0", "-C", "16", "-L", "last"});

        // Put the dataset into the filter and use filter
        removeExtreme.setInputFormat(dataset);
        Instances removeExtremeData = Filter.useFilter(dataset, removeExtreme);

        // Write a new dataset after remove extreme values
        saver.setInstances(removeExtremeData);
        saver.setFile(new File("./code/data/extremeRemoved-HepatitisCdata.arff"));
        saver.writeBatch();

        /*
         * Remove dupplicate values
         */
        // Load InterQuartile Range values dataset
        src = new DataSource("./code/data/extremeRemoved-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Create an object to remove outlier values
        RemoveDuplicates removeDuplicates = new RemoveDuplicates();

        // Put the dataset into the filter and use filter
        removeDuplicates.setInputFormat(dataset);
        Instances removeDups = Filter.useFilter(dataset, removeDuplicates);

        // Write a new dataset after remove outlier values
        saver.setInstances(removeDups);
        saver.setFile(new File("./code/data/duplicateRemoved-HepatitisCdata.arff"));
        saver.writeBatch();

        /*
         * Remove unnecessary attributes
         */
        src = new DataSource("./code/data/duplicateRemoved-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Create an object to remove attribute
        Remove remove = new Remove();

        // Set the filter option
        remove.setOptions(new String[]{"-R", "1,15,16"});

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
        src = new DataSource("./code/data/cleaned-HepatitisCdata.arff");
        dataset = src.getDataSet();

        // Create Discretize object
        Discretize discretize = new Discretize();

        // Set the filter option
        discretize.setOptions(new String[]{"-O", "-B", "10", "-M", "-1.0", "-R", "first-last", "-precision", "6"});

        // Put the dataset into the filter and use filter
        discretize.setInputFormat(dataset);
        newData = Filter.useFilter(dataset, discretize);

        // Write a new dataset after discretize
        saver.setInstances(newData);
        saver.setFile(new File("./code/data/discretized-HepatitisCdata.arff"));
        saver.writeBatch();

        /*
         * Correlation Analysis
         */
        CorrelationAttributeEval cEval = new CorrelationAttributeEval();
        for (int i = 0; i <= dataset.numAttributes() - 1; i++) {
            dataset.setClassIndex(i);
            cEval.buildEvaluator(dataset);
            for (int j = 0; j <= i; j++) {
                if (j == 0) {
                    System.out.print(
                            "A[" + i + "]" + "\t" + String.format("%.2f", cEval.evaluateAttribute(j)) + "\t"
                            );
                } else {
                    System.out.print(String.format("%.2f", cEval.evaluateAttribute(j)) + "\t");
                }
            }
            System.out.println("\t");
        }
        for (int i = 0; i <= dataset.numAttributes() - 1; i++) {
            System.out.print("\t" + "A[" + i + "]" + " ");
        }
    }
}

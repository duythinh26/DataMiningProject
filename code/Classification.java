import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Classification {
    public static String path = "./code/data/cleaned-HepatitisCdata.arff"; // file path
    private static DataSource source; // datasource
    private static Instances data; // dataset
    private static Evaluation eval; // evaluation factor

    public static void main(String[] args) throws Exception {
        // naiveBayesClassifier();
        zeroRClassifier();
    }

    /**
     * This function is used to do Naive Bayes classifier
     * build and print a 10-fold cross validation Naive Bayes model and print out
     * its evaluation result
     */
    private static void naiveBayesClassifier() throws Exception {
        // load dataset
        source = new DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(0);

        // create new Navive Bayes model
        NaiveBayes model = new NaiveBayes();
        // assign option to generate model
        String option[] = new String[] { "-K" };
        model.setOptions(option);

        // evaluation data with 10-fold cross validation Naive Bayes classifier model
        eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));
        model.buildClassifier(data);

        // print out Naive Bayes model
        System.out.println("=== Naive Bayes Model ===\n");
        System.out.println(model);
        // print out evaluation result and confusion matrix
        printConfusionMatrix();
    }
    
    private static void j48Classifier() throws Exception{
        // Load the dataset
        source = new DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(0); // Set the class attribute

        // Perform feature selection using CFS and GreedyStepwise
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        selector.setEvaluator(evaluator);
        selector.setSearch(search);
        selector.SelectAttributes(data);
        Instances selectedData = selector.reduceDimensionality(data); // Get the new Instances object with selected
                                                                      // attributes

        // Use J48 as the classifier
        Classifier j48Classifier = new J48();

        // Train the J48 classifier on the selected features
        j48Classifier.buildClassifier(selectedData);

        // Evaluate the J48 classifier
        eval = new Evaluation(selectedData);
        eval.crossValidateModel(j48Classifier, selectedData, 10, new java.util.Random(1)); // 10-fold cross validation

        // Print out J48 model
        System.out.println("=== J48 Model ===\n");
        System.out.println(j48Classifier);

        // Print out evaluation result and confusion matrix
        printConfusionMatrix();
    }  
    private static void zeroRClassifier() throws Exception{
        // load dataset
        source = new DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(0);

        // create new ZeroR baseline model
        ZeroR baseline = new ZeroR();

        // evaluation data with 10-fold cross validation ZeroR baseline classifier model
        eval = new Evaluation(data);
        eval.crossValidateModel(baseline, data, 10, new Random(1));
        baseline.buildClassifier(data);

        // print out ZeroR model
        System.out.println("=== ZeroR Model ===\n");
        System.out.println(baseline);

        // print out evaluation result and confusion matrix
        printConfusionMatrix();
    }

    /**
     * This function is used to print out evaluation results of a model and
     * confusion matrix
     **/
    private static void printConfusionMatrix() throws Exception {
        // System.out.println();
        // Print out confusion matrix
        System.out.println(eval.toMatrixString("=== Confusion matrix for fold ===\n"));
        // Print out evaluation results
        System.out.println("Correct % = " + eval.pctCorrect());
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("fMeasure = " + eval.fMeasure(1));
        System.out.println("*********");
        System.out.println(eval.toSummaryString());
    }
}

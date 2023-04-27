import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Classification {
    public static String path = "./code/data/test-HepatitisCdata.arff"; // file path
    private static DataSource source; // datasource
    private static Instances data; // dataset
    private static Evaluation eval; // evaluation factor

    public static void main(String[] args) throws Exception {
        naiveBayesClassifier();
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
        data.setClassIndex(data.numAttributes() - 1);

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


    /**
     * This function is used to print out evaluation results of a model and
     * confusion matrix
     **/
    private static void printConfusionMatrix() throws Exception {
        System.out.println();
        // Print out confusion matrix
        System.out.println(eval.toMatrixString("=== Confusion matrix for fold ===\n"));
        // Print out evaluation results
        System.out.println("Correct % = " + eval.pctCorrect());
        // System.out.println("Incorrect % = "+eval.pctIncorrect());
        // System.out.println("AUC = "+eval.areaUnderROC(1));
        // System.out.println("kappa = "+eval.kappa());
        // System.out.println("MAE = "+eval.meanAbsoluteError());
        // System.out.println("RMSE = "+eval.rootMeanSquaredError());
        // System.out.println("RAE = "+eval.relativeAbsoluteError());
        // System.out.println("RRSE = "+eval.rootRelativeSquaredError());
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("fMeasure = " + eval.fMeasure(1));
        // System.out.println("Error Rate = "+eval.errorRate());
        System.out.println();
    }
}
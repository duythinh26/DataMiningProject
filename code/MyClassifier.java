import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class MyClassifier {
    public static void main(String[] args) throws Exception {
        // Load the dataset
        DataSource source = new DataSource("./code/data/cleaned-HepatitisCdata.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(0); // Set the class attribute

        // Perform feature selection using CFS and GreedyStepwise
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        selector.setEvaluator(evaluator);
        selector.setSearch(search);
        selector.SelectAttributes(data);
        Instances selectedData = selector.reduceDimensionality(data); // Get the new Instances object with selected attributes

        // Use J48 as the classifier
        Classifier classifier = new J48();

        // Train the classifier on the selected features
        classifier.buildClassifier(selectedData);

        // Evaluate the classifier
        Evaluation eval = new Evaluation(selectedData);
        eval.crossValidateModel(classifier, selectedData, 10, new java.util.Random(1)); // 10-fold cross validation
        System.out.println(eval.toSummaryString());
    }
}

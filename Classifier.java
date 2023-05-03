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
        DataSource source = new DataSource("./code/data/HepatitisCdata.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); // Set the class attribute

        // Perform feature selection using CFS and GreedyStepwise
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        selector.setEvaluator(evaluator);
        selector.setSearch(search);
        selector.SelectAttributes(data);
        int[] indices = selector.selectedAttributes();

        // Use J48 as the classifier
        Classifier classifier = new J48();

        // Train the classifier on the selected features
        Instances selectedData = new Instances(data);
        selectedData.deleteAttributes(indices);
        classifier.buildClassifier(selectedData);

        // Evaluate the classifier
        Evaluation eval = new Evaluation(selectedData);
        eval.crossValidateModel(classifier, selectedData, 10, new java.util.Random(1)); // 10-fold cross validation
        System.out.println(eval.toSummaryString());
    }
}

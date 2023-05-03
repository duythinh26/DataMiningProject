import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

public class J48Classifier {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("./code/data/HepatitisCdata.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        J48 j48 = new J48();
        
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(j48, data, 10, new Random(1));
        System.out.println(eval.toSummaryString());
        j48.buildClassifier(data);
        
        Instances test = 
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = j48.classifyInstance(test.instance(i));
            System.out.println("Prediction for instance " + i + ": " + data.classAttribute().value((int) pred));
        }
    }
}

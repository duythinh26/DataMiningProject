import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Folds {
    public static void main(String args[]) throws Exception{
        //load dataset
        DataSource source = new DataSource("./code/data/cleaned-HepatitisCdata.arff");
        Instances dataset = source.getDataSet();
        //set class index to the last attribute
        dataset.setClassIndex(0);

        //create the classifier
        //choose the executed algorithm
        NaiveBayes model = new NaiveBayes();
        // J48 model = new J48();
        // ZeroR model = new ZeroR();

        int seed = 1;
        int folds = 10;
        // randomize data
        Random rand = new Random(seed);
        //create random dataset
        Instances randData = new Instances(dataset);
        randData.randomize(rand);
        //stratify
        if (randData.classAttribute().isNominal())
            randData.stratify(folds);

        // perform cross-validation
        for (int n = 0; n < folds; n++) {
            Evaluation eval = new Evaluation(randData);
            //get the folds
            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);
            // build and evaluate classifier
            model.buildClassifier(train);
            eval.evaluateModel(model, test);

            // output evaluation
            System.out.println();
            System.out.println(eval.toMatrixString("**** Confusion matrix for fold " + (n+1) + "/" + folds + " ****\n"));
            System.out.println("Correct % = "+eval.pctCorrect());
            System.out.println("Precision = "+eval.precision(1));
            System.out.println("Recall = "+eval.recall(1));
            System.out.println("fMeasure = "+eval.fMeasure(1));
        }
    }
}

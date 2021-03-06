package test.utils;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;

import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Converts Mallet instanceList to Weka ARFF/Instances
 * 
 * @author Syeed Ibn Faiz
 */
public class Converter {
	/**
	 * Converts Mallet InstanceList into Weka ARFF format
	 * 
	 * @param instances
	 *            Mallet instances
	 * @param description
	 *            a String description required by Weka
	 * @return ARFF representation of the InstanceList
	 */
	public static String convert2ARFF(InstanceList instances, String description) {
		Alphabet dataAlphabet = instances.getDataAlphabet();
		Alphabet targetAlphabet = instances.getTargetAlphabet();
		StringBuilder sb = new StringBuilder();
		sb.append("@Relation \"").append(description).append("\"\n\n");
		int size = dataAlphabet.size();
		for (int i = 0; i < size; i++) {
			sb.append("@attribute \"").append(dataAlphabet.lookupObject(i).toString().replaceAll("\\s+", "_"))
					.append("_").append(i);
			sb.append("\" numeric\n");
		}
		sb.append("@attribute target {");
		for (int i = 0; i < targetAlphabet.size(); i++) {
			if (i != 0)
				sb.append(",");
			sb.append(targetAlphabet.lookupObject(i).toString().replace(",", ";"));
		}
		sb.append("}\n\n@data\n");
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			sb.append("{");
			FeatureVector fv = (FeatureVector) instance.getData();
			int[] indices = fv.getIndices();
			double[] values = fv.getValues();
			boolean[] attrFlag = new boolean[size];
			double[] attrValue = new double[size];
			for (int j = 0; j < indices.length; j++) {
				attrFlag[indices[j]] = true;
				attrValue[indices[j]] = values[j];
			}
			for (int j = 0; j < attrFlag.length; j++) {
				if (attrFlag[j]) {
					// sb.append(j).append(" 1, ");
					sb.append(j).append(" ").append(attrValue[j]).append(", ");
				}
			}
			sb.append(attrFlag.length).append(" ").append(instance.getTarget().toString().replace(",", ";"));
			sb.append("}\n");
		}
		return sb.toString();
	}

	/**
	 * Converts Mallet InstanceList into Weka Instances
	 * 
	 * @param instanceList
	 * @return
	 * @throws IOException
	 */
	public static Instances convert2WekaInstances(InstanceList instanceList) throws IOException {
		String arff = convert2ARFF(instanceList, "DESC");
		StringReader reader = new StringReader(arff);
		Instances instances = new Instances(reader);
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
	}
	
	public static void main(String[] args) throws IOException, Exception {  
	     ArrayList<Pipe> pipes = new ArrayList<Pipe>();  
	     pipes.add(new Target2Label());  
	     pipes.add(new CharSequence2TokenSequence());  
	     pipes.add(new TokenSequence2FeatureSequence());  
	     pipes.add(new FeatureSequence2FeatureVector());  
	     SerialPipes pipe = new SerialPipes(pipes);  
	     //prepare training instances  
	     InstanceList trainingInstanceList = new InstanceList(pipe);  
	     trainingInstanceList.addThruPipe(new CsvIterator(new FileReader("webkb-train-stemmed.txt"),  
	         "(.*)\t(.*)", 2, 1, -1));  
	     //prepare test instances  
	     InstanceList testingInstanceList = new InstanceList(pipe);  
	     testingInstanceList.addThruPipe(new CsvIterator(new FileReader("webkb-test-stemmed.txt"),  
	         "(.*)\t(.*)", 2, 1, -1));  
	     //Using a classifier in Mallet  
	     ClassifierTrainer trainer = new NaiveBayesTrainer();  
	     Classifier classifier = trainer.train(trainingInstanceList);  
	     System.out.println("Accuracy[Mallet]: " + classifier.getAccuracy(testingInstanceList));  
	     //Getting Weka Instances  
	     Instances trainingInstances = Converter.convert2WekaInstances(trainingInstanceList);  
	     Instances testingInstances = Converter.convert2WekaInstances(testingInstanceList);  
	     //Using a classifier in Weka  
	     NaiveBayesMultinomial naiveBayesMultinomial = new NaiveBayesMultinomial();  
	     naiveBayesMultinomial.buildClassifier(trainingInstances);  
	     Evaluation evaluation = new Evaluation(testingInstances);  
	     evaluation.evaluateModel(naiveBayesMultinomial, testingInstances);  
	     System.out.println("Accuracy[Weka]: " + evaluation.correct() / testingInstanceList.size());      
	   }  
}
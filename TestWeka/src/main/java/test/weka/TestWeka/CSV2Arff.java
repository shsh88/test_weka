package test.weka.TestWeka;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class CSV2Arff {
	/**
	 * takes 2 arguments: - CSV input file - ARFF output file
	 */
	public static void main(String[] args) throws Exception {

		// load CSV
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("resources/train_stances_noid.csv"));
		//loader.setSource(new File("resources/train_stances.csv"));
		Instances data = loader.getDataSet();

		// save ARFF
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("resources/train_stances_noid.arff"));
		saver.setDestination(new File("resources/train_stances_noid.arff"));
		//saver.setFile(new File("resources/train_stances.arff"));
		//saver.setDestination(new File("resources/train_stances.arff"));
		saver.writeBatch();
	}
}

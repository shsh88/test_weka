package test.weka.TestWeka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ArffReader {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("resources/test_bin.arff");
		Instances data = source.getDataSet();

		int numInst = data.numInstances();
		int numAtt = data.numAttributes();

		for (int i = 0; i < 20; i++) {
			String title = data.get(i).stringValue(0);
			String body = data.get(i).stringValue(1);
			
			System.out.println("( " + i + " )");
			System.out.println("title: " + title);
			System.out.println("body: " + body);
			
			System.out.println();

		}
	}

}

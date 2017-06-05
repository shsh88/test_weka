package test.weka.TestWeka;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class TwoClassStanceDetection {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			System.out.println("File not found" + filename);
		}
		return inputReader;

	}

	public static void main(String[] args) throws Exception {
		
		DataSource headerSource = new DataSource("resources/headers.arff");
		Instances headerData = headerSource.getDataSet();
		
		StringToWordVector headerFilter = new StringToWordVector();
		headerFilter.setAttributeIndices("first");
		
		headerFilter.setDoNotOperateOnPerClassBasis(true);
		headerFilter.setDictionaryFileToSaveTo(new File("resources/dic"));
		headerFilter.setTFTransform(true);
		headerFilter.setInputFormat(headerData);
		
		Instances headerFiltered = Filter.useFilter(headerData, headerFilter);
		
		 System.out.println("\n\nFiltered data:\n\n" + headerFiltered);
		

	}

}

package test.weka.TestWeka;

import weka.core.Attribute;
import weka.core.DenseInstance;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.opencsv.CSVReader;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * In this class we read the data from the 2 provided .csv data files
 * "train_bodies.csv" and "train_stances.csv" and create an ARFF file out of
 * them
 * 
 * example used: https://weka.wikispaces.com/Creating+an+ARFF+file
 * 
 * @author razan
 *
 */

public class ProcessCSVData {

	public static HashMap<Integer, String> getIdBodyMap(String File) {
		HashMap<Integer, String> bodyMap = new HashMap<>(100, 100);
		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(File));
			String[] line;
			line = reader.readNext();
			while ((line = reader.readNext()) != null) {
				bodyMap.put(new Integer(line[0]), line[1]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return bodyMap;
	}

	public static void main(String[] args) throws IOException {
		String bodiesCSV = "resources/train_bodies.csv";
		String stancesCSV = "resources/train_stances1.csv";

		ArrayList<Attribute> attributes;
		List<String> attributeValues;
		String stances[] = new String[] { "unrelated", "related" };
		List<String> stanceValues = Arrays.asList(stances);
		Instances data;

		// 1.Setup attributes
		attributes = new ArrayList<>();
		attributes.add(new Attribute("Title", (ArrayList<String>) null));
		attributes.add(new Attribute("Body", (ArrayList<String>) null));
		attributes.add(new Attribute("Class", stanceValues));

		// 2. create Instances object
		data = new Instances("title_body_stance_relation_bin", attributes, 1000);

		// 3. Fill the data

		HashMap<Integer, String> bodyMap = getIdBodyMap(bodiesCSV);

		CSVReader stancesReader = null;
		try {
			stancesReader = new CSVReader(new FileReader(stancesCSV));
			String[] stancesline;
			stancesReader.readNext();
			while ((stancesline = stancesReader.readNext()) != null) {
				double values[] = new double[data.numAttributes()];
				values[0] = data.attribute(0).addStringValue(stancesline[0]);
				values[1] = data.attribute(1).addStringValue(bodyMap.get(Integer.valueOf(stancesline[1])));
				values[2] = stanceValues.indexOf(stancesline[2]);
				data.add(new DenseInstance(1.0, values));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		// 4. output data
		//System.out.println(data);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new java.io.File("resources/test_bin.arff"));
		// saver.setDestination(new File("resources/train_stances_noid.arff"));
		saver.writeBatch();
	}

}

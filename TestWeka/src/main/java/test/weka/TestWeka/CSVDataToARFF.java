package test.weka.TestWeka;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import com.opencsv.CSVReader;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class CSVDataToARFF {

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

		ArrayList<Attribute> attributesHeaders;
		ArrayList<Attribute> attributesBodies;
		ArrayList<Attribute> finalAttributes;
		List<String> attributeValues;
		String stances[] = new String[] { "unrelated", "related" };
		List<String> stanceValues = Arrays.asList(stances);
		Instances HeaderData;
		Instances bodyData;
		Instances finalInstances;

		// 1.Setup attributes
		attributesHeaders = new ArrayList<>();
		attributesBodies = new ArrayList<>();
		finalAttributes = new ArrayList<>();

		attributesHeaders.add(new Attribute("Title", (ArrayList<String>) null));
		attributesHeaders.add(new Attribute("Bodey_ID"));

		attributesBodies.add(new Attribute("ID"));
		attributesBodies.add(new Attribute("Body", (ArrayList<String>) null));

		finalAttributes.add(new Attribute("similarity"));
		finalAttributes.add(new Attribute("Class", stanceValues));

		// 2. create Instances object
		HeaderData = new Instances("header_relation", attributesHeaders, 1000);
		bodyData = new Instances("body_relation", attributesBodies, 1000);
		finalInstances = new Instances("final_relation", finalAttributes, 1000);

		HashMap<Integer, String> bodyMap = getIdBodyMap(bodiesCSV);

		CSVReader stancesReader = null;
		try {
			stancesReader = new CSVReader(new FileReader(stancesCSV));
			String[] stancesline;
			stancesReader.readNext();
			while ((stancesline = stancesReader.readNext()) != null) {
				double headerValues[] = new double[HeaderData.numAttributes()];
				headerValues[0] = HeaderData.attribute(0).addStringValue(stancesline[0]);
				headerValues[1] = Integer.valueOf(stancesline[1]);
				HeaderData.add(new DenseInstance(1.0, headerValues));

				// double finalValues[] = new
				// double[finalInstances.numAttributes()];
				// finalValues[0] = 1.0;
				// finalValues[1] = stanceValues.indexOf(stancesline[2]);
				// finalInstances.add(new DenseInstance(1.0, finalValues));

			}

			stancesReader.close();

			CSVReader bodyReader = null;
			String[] bodyline;
			bodyReader = new CSVReader(new FileReader(bodiesCSV));
			bodyReader.readNext();
			while ((bodyline = bodyReader.readNext()) != null) {

				double bodyValues[] = new double[bodyData.numAttributes()];
				bodyValues[0] = Integer.valueOf(bodyline[0]);
				bodyValues[1] = bodyData.attribute(1).addStringValue(bodyline[1]);
				bodyData.add(new DenseInstance(1.0, bodyValues));
			}
			bodyReader.close();

		} catch (IOException e) {
			e.printStackTrace();
		}

		// 4. output data
		// System.out.println(data);

		ArffSaver headerSaver = new ArffSaver();
		headerSaver.setInstances(HeaderData);
		headerSaver.setFile(new java.io.File("resources/headers.arff"));
		headerSaver.writeBatch();
		
		ArffSaver bodySaver = new ArffSaver();
		bodySaver.setInstances(bodyData);
		bodySaver.setFile(new java.io.File("resources/bodies.arff"));
		bodySaver.writeBatch();

	}

}

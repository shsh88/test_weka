package test.fnc.classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import com.opencsv.CSVReader;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;

public class FNCUtility {

	public static List<List<String>> readStances(String filePath) throws FileNotFoundException, IOException {
		CSVReader stancesReader = new CSVReader(new FileReader(filePath));
		String[] stancesline;
		List<List<String>> stances = new ArrayList<>();
		stancesReader.readNext();
		while ((stancesline = stancesReader.readNext()) != null) {
			List<String> record = new ArrayList<>();
			for (int i = 0; i < stancesline.length; i++)
				record.add(stancesline[i]);
			stances.add(record);
		}
		return stances;
	}

	public static List<List<String>> readTestStances(String filePath) throws FileNotFoundException, IOException {
		CSVReader stancesReader = new CSVReader(new FileReader(filePath));
		String[] stancesline;
		List<List<String>> stances = new ArrayList<>();
		stancesReader.readNext();
		while ((stancesline = stancesReader.readNext()) != null) {
			List<String> record = new ArrayList<>();
			record.add(stancesline[0]);
			record.add(stancesline[1]);
			stances.add(record);
		}
		stancesReader.close();
		return stances;
	}

	public static HashMap<Integer, String> getBodiesMap(String bodiesFilePath) {
		HashMap<Integer, String> bodyMap = new HashMap<>(100, 100);
		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(bodiesFilePath));
			String[] line;
			line = reader.readNext();
			while ((line = reader.readNext()) != null) {
				bodyMap.put(Integer.valueOf(line[0]), line[1]);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return bodyMap;
	}

	/**
	 * Loads an ARFF file into an instances object.
	 * 
	 * @param fileName
	 *            The name of the file to be loaded.
	 * @param classIndex
	 */
	public static Instances loadARFF(String fileName, int classIndex) {
		Instances inputInstances = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			inputInstances = arff.getData();
			inputInstances.setClassIndex(classIndex);
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		} catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
		return inputInstances;
	}

	static void saveInstancesToARFFFile(String filepath, Instances instances) throws IOException {
		System.out.println(instances.size());
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new java.io.File(filepath));
		saver.writeBatch();
	}

}

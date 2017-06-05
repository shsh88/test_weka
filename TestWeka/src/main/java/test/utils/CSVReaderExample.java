package test.utils;

import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class CSVReaderExample {

	public static void main(String[] args) {

		String csvFile = "resources/train_bodies.csv";

		Map<Integer, String> bodyMap = new HashMap<>(100, 100);

		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(csvFile));
			String[] line;
			line = reader.readNext();
			while ((line = reader.readNext()) != null) {
				System.out.println("Body [id= " + line[0] + ", articleBody= " + line[1] + "]");
				bodyMap.put(new Integer(line[0]), line[1]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println(bodyMap.size());
	}

}

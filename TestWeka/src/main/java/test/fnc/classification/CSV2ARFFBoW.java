package test.fnc.classification;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.stopwords.MultiStopwords;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class CSV2ARFFBoW {

	private static Map<Integer, String> bodiesMap;
	private static List<List<String>> stances;
	static String stancesClasses[] = new String[] { "agree", "disagree", "discuss", "unrelated" };
	private static Instances instances;
	private static Instances inputInstances;

	public static Instances getInstances(String stancesFilePath, String bodiesFilePath)
			throws FileNotFoundException, IOException {
		bodiesMap = FNCUtility.getBodiesMap(bodiesFilePath);
		stances = FNCUtility.readStances(stancesFilePath);

		ArrayList<Attribute> attributes = new ArrayList<>();
		attributes.add(new Attribute("Title", (ArrayList<String>) null));
		attributes.add(new Attribute("Body", (ArrayList<String>) null));

		List<String> stanceValues = Arrays.asList(stancesClasses);
		attributes.add(new Attribute("class", stanceValues));

		// System.out.println(attributes.size());
		Instances instances = new Instances("fnc", attributes, 1000);
		for (int i = 0; i < stances.size(); i++) {
			
			if ((i + 1) % 1000 == 0)
				System.out.println("Processing after i " + i);

			List<Integer> features = new ArrayList<>();

			String headline = getLemmatizedInput(stances.get(i).get(0));
			features.add(instances.attribute(0).addStringValue(headline));
			String body = getLemmatizedInput(bodiesMap.get(Integer.valueOf(stances.get(i).get(1))));
			features.add(instances.attribute(1).addStringValue(body));
			features.add(stanceValues.indexOf(stances.get(i).get(2)));

			double[] a = new double[features.size()];
			int j = 0;
			for (double feature : features) {
				a[j++] = feature;
			}

			// System.out.println(features.size());
			instances.add(new DenseInstance(1.0, a));
		}

		return instances;

	}

	private static Instances outputInstances;

	/**
	 * Index the inputInstances string features using the StringToWordVector
	 * filter.
	 */
	public static void index() {
		// outputInstances = inputInstances;
		try {

			// Set the tokenizer
			NGramTokenizer tokenizer = new NGramTokenizer();
			tokenizer.setNGramMinSize(1);
			tokenizer.setNGramMaxSize(1);
			tokenizer.setDelimiters("\\W");

			WordTokenizer wTokinizer = new WordTokenizer();
			// wTokinizer.setDelimiters("\r\n\t\\");

			// Set the filter
			StringToWordVector filter = new StringToWordVector();
			filter.setTokenizer(wTokinizer);
			filter.setAttributeIndicesArray(new int[] { 0, 1 });
			filter.setInputFormat(inputInstances);
			// filter.setWordsToKeep(1000);
			filter.setDoNotOperateOnPerClassBasis(true);
			filter.setLowerCaseTokens(true);
			filter.setOutputWordCounts(true);
			filter.setStopwordsHandler(new Rainbow());
			filter.setDictionaryFileToSaveTo(new File("resources/dic_nostop_lemma.txt"));

			// Filter the input instances into the output ones
			outputInstances = Filter.useFilter(inputInstances, filter);

			System.out.println("===== Filtering dataset done =====");
		} catch (Exception e) {
			System.out.println("Problem found when training");
			e.printStackTrace();
		}
	}

	private static String getLemmatizedInput(String txt) {
		String words = "";
		StanfordLemmatizer lemmatizer = new StanfordLemmatizer();
		List<String> lemmas = lemmatizer.lemmatize(txt);
		for (String lemma : lemmas) {
			words = words.concat(lemma);
			words = words.concat(" ");
		}

		return words.trim();
	}

	/**
	 * Save an instances object into an ARFF file.
	 * 
	 * @param fileName
	 *            The name of the file to be saved.
	 */
	public static void saveARFF(String fileName) {

		try {
			PrintWriter writer = new PrintWriter(new FileWriter(fileName));
			writer.print(outputInstances);
			System.out.println("===== Saved dataset: " + fileName + " =====");
			writer.close();
		} catch (IOException e) {
			System.out.println("Problem found when writing: " + fileName);
		}
	}

	public static void main(String[] args) throws Exception {

		instances = getInstances("resources/train_stances.csv", "resources/train_bodies.csv");
		FNCUtility.saveInstancesToARFFFile("resources/title_body_stance.arff", instances);

		inputInstances = FNCUtility.loadARFF("resources/title_body_stance.arff");

		index();
		saveARFF("resources/BoW_title_body_stance_lemma.arff");
		/*
		 * String txt =
		 * "How could you be seeing into my eyes like open doors? \n"+
		 * "You led me down into my core where I've became so numb \n"+
		 * "Without a soul my spirit's sleeping somewhere cold \n"+
		 * "Until you find it there and led it back home \n"+
		 * "You woke me up inside \n"+
		 * "Called my name and saved me from the dark \n"+
		 * "You have bidden my blood and it ran \n"+
		 * "Before I would become undone \n"+
		 * "You saved me from the nothing I've almost become \n"+
		 * "You were bringing me to life \n"+
		 * "Now that I knew what I'm without \n"+ "You can've just left me \n"+
		 * "You breathed into me and made me real \n"+
		 * "Frozen inside without your touch \n"+
		 * "Without your love, darling \n"+
		 * "Only you are the life among the dead \n"+
		 * "I've been living a lie, there's nothing inside \n"+
		 * "You were bringing me to life.";
		 * System.out.println(getLemmatizedInput(txt));
		 */
	}

}

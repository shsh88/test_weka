package test.fnc.classification;

import static com.google.common.base.Predicates.in;
import static org.simmetrics.builders.StringDistanceBuilder.with;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;
import java.util.Map.Entry;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.simmetrics.StringDistance;
import org.simmetrics.metrics.CosineSimilarity;
import org.simmetrics.simplifiers.Simplifiers;
import org.simmetrics.tokenizers.Tokenizers;

import com.google.common.base.Predicates;
import com.google.common.collect.Sets;
import com.opencsv.CSVReader;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class FNCTestFeatureEngineering {

	private static TreeSet<String> stopSet;

	private static HashMap<Integer, String> idBodyMap;

	private static List<List<String>> stances;

	private static ParagraphVectors pvecs;

	private static String[] refutingWords = { "fake", "fraud", "hoax", "false", "deny", "denies", "not", "despite",
			"nope", "doubt", "doubts", "bogus", "debunk", "pranks", "retract" };
	private static String[] discussWords = { "according", "maybe", "reporting", "reports", "say", "says", "claim",
			"claims", "purportedly", "investigating", "told", "tells", "allegedly", "validate", "verify" };

	private static StringDistance metric;
	private static Instances instances;

	//private static List<List<String>> titleBodyPairs;

	private static Map<String, String> titleIdMap;

	public static void setData() throws IOException {
		File bodiesFile = new File("resources/test_bodies.csv");

		idBodyMap = getBodiesMap(bodiesFile);

		readStances();
	}

	private static void readStances() throws FileNotFoundException, IOException {
		CSVReader stancesReader = new CSVReader(new FileReader("resources/competition_test_stances.csv"));
		String[] stancesline;
		stances = new ArrayList<>();
		stancesReader.readNext();
		while ((stancesline = stancesReader.readNext()) != null) {
			List<String> record = new ArrayList<>();
			record.add(stancesline[0]);
			record.add(stancesline[1]);
			record.add(stancesline[2]);
			stances.add(record);
		}
		stancesReader.close();
	}

	private static Instances getWekaInstances() {
		ArrayList<Attribute> attributes = new ArrayList<>();

		for (int i = 0; i < pvecs.getLayerSize(); i++) {
			attributes.add(new Attribute("titleVec_"+i));
		}

		for (int i = 0; i < pvecs.getLayerSize(); i++) {
			attributes.add(new Attribute("bodyVec_"+i));
		}
		
		attributes.add(new Attribute("word_overlap"));

		for (String refute : refutingWords) {
			attributes.add(new Attribute("refute_" + refute));
		}

		for (String refute : discussWords) {
			attributes.add(new Attribute("discuss_" + refute));
		}

		attributes.add(new Attribute("pol_head"));
		attributes.add(new Attribute("pol_body"));

		int[] cgramSizes = { 2, 8, 4, 16 };
		for (int size : cgramSizes) {
			attributes.add(new Attribute("cgram_hits_" + size));
			attributes.add(new Attribute("cgram_early_hits_" + size));
			attributes.add(new Attribute("cgram_first_hits_" + size));
			attributes.add(new Attribute("cgram_tail_hits_" + size));
		}

		int[] ngramSizes = { 2, 3, 4, 5, 6 };
		for (int size : ngramSizes) {
			attributes.add(new Attribute("ngram_hits_" + size));
			attributes.add(new Attribute("ngram_early_hits_" + size));
			attributes.add(new Attribute("ngram_tail_hits" + size));
		}

		attributes.add(new Attribute("bin_co_occ_count"));
		attributes.add(new Attribute("bin_co_occ_early"));

		attributes.add(new Attribute("bin_co_occ_stop_count"));
		attributes.add(new Attribute("bin_co_occ_stop_early"));

		attributes.add(new Attribute("similarity"));

		String stancesClasses[] = new String[] { "agree", "disagree", "discuss", "unrelated" };
		List<String> stanceValues = Arrays.asList(stancesClasses);
		attributes.add(new Attribute("class", stanceValues));

		instances = new Instances("fnc", attributes, 1000);
		for (int i = 0; i < stances.size(); i++) {
			// for (int i = 0; i < 500; i++) {
			List<Double> features = new ArrayList<>();

			String headline = stances.get(i).get(0);
			String body = idBodyMap.get(Integer.valueOf(stances.get(i).get(1)));
			
			features.addAll(getTitleAndBodyVecs(headline, stances.get(i).get(1), i));

			features.add(getWordOverlapFeature(headline, body));
			features.addAll(getRefutingFeature(headline, body));
			features.addAll(getDiscussFeature(headline, body));
			features.addAll(getPolarityFeatures(headline, body));
			features.addAll(countGrams(headline, body));
			features.addAll(binaryCoOccurenceFeatures(headline, body));
			features.addAll(binaryCoOccurenceStopFeatures(headline, body));
			features.add(getSimilarityFeature(headline, stances.get(i).get(1), i));
			features.add((double) stanceValues.indexOf(stances.get(i).get(2)));

			double[] a = new double[features.size()];
			int j = 0;
			for (double feature : features) {
				a[j++] = feature;
			}

			instances.add(new DenseInstance(1.0, a));
		}

		return instances;
	}

	private static List<Double> getTitleAndBodyVecs(String headline, String bodyId, int record) {
		List<Double> titleBodyVec = new ArrayList<>();
		System.out.println(record + " : " + titleIdMap.get(headline));
		INDArray titleVec = pvecs.getLookupTable().vector(titleIdMap.get(headline)); // get the text vector by the id / label
		for(int k = 0; k < pvecs.getLayerSize(); k++){ //line 1383 in the example
			titleBodyVec.add(titleVec.getDouble(k)); //feature_k of body 0
		}
		
		INDArray bodyVec = pvecs.getLookupTable().vector(bodyId); // get the text vector by the id / label
		for(int k = 0; k < pvecs.getLayerSize(); k++){ //line 1383 in the example
			titleBodyVec.add(bodyVec.getDouble(k)); //feature_k of body 0
		}
		
		
		return titleBodyVec;
	}

	private static void saveInstancesToARFFFile(String filepath) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new java.io.File(filepath));
		saver.writeBatch();
	}

	private static HashMap<Integer, String> getBodiesMap(File bodiesFile) {
		HashMap<Integer, String> bodyMap = new HashMap<>(100, 100);
		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(bodiesFile));
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

	public static String cleanText(String text) {
		text = text.replaceAll("[^\\p{ASCII}]", "");
		text = text.replaceAll("\\s+", " ");
		text = text.replaceAll("\\p{Cntrl}", "");
		text = text.replaceAll("[^\\p{Print}]", "");
		text = text.replaceAll("\\p{C}", "");
		return text;
	}

	public List<String> tokinize(String sourceText, String modelPath) throws InvalidFormatException, IOException {
		InputStream modelIn = null;
		modelIn = new FileInputStream(modelPath);

		TokenizerModel model = new TokenizerModel(modelIn);

		Tokenizer tokenizer = new TokenizerME(model);
		String tokens[] = tokenizer.tokenize(sourceText);
		return Arrays.asList(tokens);
	}

	public static List<String> lemmatize(String text) {
		return new Lemmatizer().lemmatize(text);
	}

	public static void initializeStopwords(String stopFile) throws Exception {
		stopSet = new TreeSet<>();
		Scanner s = new Scanner(new FileReader(stopFile));
		while (s.hasNext())
			stopSet.add(s.next());
		s.close();
	}

	public static boolean isStopword(String word) {
		if (word.length() < 2)
			return true;
		if (word.charAt(0) >= '0' && word.charAt(0) <= '9')
			return true; // remove numbers, "23rd", etc
		if (stopSet.contains(word))
			return true;
		else
			return false;
	}

	public static List<String> removeStopWords(List<String> wordsList) {
		List<String> wordsNoStop = new ArrayList<>();

		for (String word : wordsList) {
			if (word.isEmpty())
				continue;
			if (isStopword(word))
				continue; // remove stopwords
			wordsNoStop.add(word);
		}
		return wordsNoStop;

	}

	// This method calculate the feature for one record
	public static double getWordOverlapFeature(String headLine, String body) {
		String cleanHeadline = cleanText(headLine);
		String cleanBody = cleanText(body);

		// System.out.println(cleanHeadline);
		// System.out.println(cleanBody);
		List<String> headLineLem = lemmatize(cleanHeadline);
		List<String> bodyLem = lemmatize(cleanBody);

		Set<String> intersectinSet = new HashSet<>(headLineLem);
		Set<String> bodySet = new HashSet<>(bodyLem);

		Set<String> UnionSet = new HashSet<>(headLineLem);

		intersectinSet.retainAll(bodySet);
		UnionSet.addAll(bodySet);

		return (double) intersectinSet.size() / (double) UnionSet.size();

	}

	public static List<Double> getRefutingFeature(String headLine, String body) {

		String cleanHeadline = cleanText(headLine);
		List<String> headLineLem = lemmatize(cleanHeadline);

		List<Double> f = new ArrayList<>();

		for (String refutingWord : refutingWords) {
			if (headLineLem.contains(refutingWord))
				f.add(1.0); // TODO maychange this to add integers
			else
				f.add(0.0);
		}
		return f;
	}

	public static List<Double> getPolarityFeatures(String headLine, String body) {
		List<String> refutingWordsList = Arrays.asList(refutingWords);

		String cleanHeadline = cleanText(headLine);
		String cleanBody = cleanText(body);
		List<String> headLineLem = lemmatize(cleanHeadline);
		List<String> bodyLem = lemmatize(cleanBody);

		List<Double> f = new ArrayList<>();

		int h = 0;
		for (String word : headLineLem) {
			if (refutingWordsList.contains(word))
				h++;
		}
		f.add((double) (h % 2));

		int b = 0;
		for (String word : bodyLem) {
			if (refutingWordsList.contains(word))
				b++;
		}
		f.add((double) (b % 2));
		return f;
	}

	public static List<Double> getDiscussFeature(String headLine, String body) {

		String cleanHeadline = cleanText(headLine);
		List<String> headLineLem = lemmatize(cleanHeadline);

		List<Double> f = new ArrayList<>();

		for (String discussWord : discussWords) {
			if (headLineLem.contains(discussWord))
				f.add(1.0); // TODO maychange this to add integers
			else
				f.add(0.0);
		}
		return f;
	}

	public static List<String> getNGrams(String text, int n) {
		List<String> ret = new ArrayList<String>();
		String[] input = text.split(" ");
		for (int i = 0; i <= input.length - n; i++) {
			String ngram = "";
			for (int j = i; j < i + n; j++)
				ngram += input[j] + " ";
			ngram.trim();
			ret.add(ngram);
		}
		return ret;
	}

	public static List<String> getCharGrams(String text, int n) {
		List<String> ret = new ArrayList<String>();
		for (int i = 0; i <= text.length() - n; i++) {
			String ngram = "";
			for (int j = i; j < i + n; j++)
				ngram += text.charAt(j);
			ngram.trim();
			ret.add(ngram);
		}
		return ret;
	}

	/**
	 * vector specifying the sum of how often character sequences of length
	 * 2,4,8,16 in the headline appear in the entire body, the first 100
	 * characters and the first 255 characters of the body.
	 * 
	 * @param headLine
	 * @param body
	 */
	public static List<Double> getCharGramsFeatures(String headLine, String body, int size) {
		List<String> h = removeStopWords(Arrays.asList(headLine.split(" ")));

		// get the string back
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		headLine = sb.toString().trim();
		List<String> grams = getCharGrams(headLine, size);

		int gramHits = 0;
		int gramEarlyHits = 0;
		int gramFirstHits = 0;
		int gramTailHits = 0;

		for (String gram : grams) {
			if (body.contains(gram)) {
				gramHits++;
			}
			if (body.length() >= 255) {
				if (body.substring(0, 255).contains(gram)) {
					gramEarlyHits++;
				}

				if (body.substring(body.length() - 255).contains(gram)) {
					gramTailHits++;
				}

			} else {
				if (body.contains(gram)) {
					gramEarlyHits++;
				}

				if (body.contains(gram)) {
					gramTailHits++;
				}
			}

			if (body.length() >= 100) {
				if (body.substring(0, 100).contains(gram)) {
					gramFirstHits++;
				}

				if (body.substring(body.length() - 100).contains(gram)) {
					gramTailHits++;
				}
			} else {
				if (body.contains(gram)) {
					gramFirstHits++;
				}
				if (body.contains(gram)) {
					gramTailHits++;
				}
			}
		}

		List<Double> f = new ArrayList<>();
		f.add((double) gramHits);
		f.add((double) gramEarlyHits);
		f.add((double) gramFirstHits);
		f.add((double) gramTailHits);

		return f;
	}

	public static List<Double> getNGramsFeatures(String headLine, String body, int size) {

		List<String> grams = getNGrams(headLine, size);

		int gramHits = 0;
		int gramEarlyHits = 0;
		int gramTailHits = 0;

		for (String gram : grams) {
			if (body.contains(gram)) {
				gramHits++;
			}
			if (body.length() >= 255) {
				if (body.substring(0, 255).contains(gram)) {
					gramEarlyHits++;
				}

				if (body.substring(body.length() - 255).contains(gram)) {
					gramTailHits++;
				}

			} else {
				if (body.contains(gram)) {
					gramEarlyHits++;
				}
				if (body.contains(gram)) { // TODO do weneed to look in this
											// case
					gramTailHits++;
				}
			}

		}

		List<Double> f = new ArrayList<>();
		f.add((double) gramHits);
		f.add((double) gramEarlyHits);
		f.add((double) gramTailHits);

		return f;

	}

	public static List<Double> binaryCoOccurenceFeatures(String headLine, String body) {
		int binCount = 0;
		int binCountEarly = 0;

		String[] cleanHeadLine = cleanText(headLine).split(" ");

		String cleanBody = cleanText(body);

		for (String token : cleanHeadLine) {
			if (cleanBody.contains(token)) // TODO traverse, won't we find this?
											// --> verse
				binCount++;
			if (cleanBody.length() >= 255) {
				if (cleanBody.substring(0, 255).contains(token))
					binCountEarly++;
			} else {
				if (cleanBody.contains(token))
					binCountEarly++;
			}
		}

		List<Double> f = new ArrayList<>();
		f.add((double) binCount);
		f.add((double) binCountEarly);
		return f;

	}

	public static List<Double> binaryCoOccurenceStopFeatures(String headLine, String body) {
		int binCount = 0;
		int binCountEarly = 0;

		String[] cleanHeadLine = cleanText(headLine).split(" ");
		List<String> cleanHead = removeStopWords(Arrays.asList(cleanHeadLine));

		String cleanBody = cleanText(body);

		for (String token : cleanHead) {
			if (cleanBody.contains(token)) // TODO traverse, won't we find this?
											// --> verse
				binCount++;
			if (cleanBody.length() >= 255) {
				if (cleanBody.substring(0, 255).contains(token))
					binCountEarly++;
			} else {
				if (cleanBody.contains(token))
					binCountEarly++;
			}
		}

		List<Double> f = new ArrayList<>();
		f.add((double) binCount);
		f.add((double) binCountEarly);
		return f;
	}

	public static List<Double> countGrams(String headLine, String body) {
		String cleanHeadline = cleanText(headLine);
		String cleanBody = cleanText(body);

		List<Double> f = new ArrayList<>();
		f.addAll(getCharGramsFeatures(cleanHeadline, cleanBody, 2));
		f.addAll(getCharGramsFeatures(cleanHeadline, cleanBody, 8));
		f.addAll(getCharGramsFeatures(cleanHeadline, cleanBody, 4));
		f.addAll(getCharGramsFeatures(cleanHeadline, cleanBody, 16));
		f.addAll(getNGramsFeatures(cleanHeadline, cleanBody, 2));
		f.addAll(getNGramsFeatures(cleanHeadline, cleanBody, 3));
		f.addAll(getNGramsFeatures(cleanHeadline, cleanBody, 4));
		f.addAll(getNGramsFeatures(cleanHeadline, cleanBody, 5));
		f.addAll(getNGramsFeatures(cleanHeadline, cleanBody, 6));
		return f;
	}

	public static Double getSimilarityFeature(String headLine, String bodyId, int record) {
		//return pvecs.similarity(titleBodyPairs.get(record).get(0), bodyId);
		//System.out.println(headLine + " : " + titleIdMap.get(headLine) + " : " + bodyId);
		return pvecs.similarity(titleIdMap.get(headLine), bodyId);
	}

	public static void FNCFeaturesToARFF(String filePath) {
		// get the data also first
	}

	public static void main(String[] args) throws Exception {
		initializeStopwords("resources/stopwords.txt");

		// String text = new FNCFeatureEngineering().cleanText("Our Retina
		// MacBook Air rumour article brings"
		// + " together everything we know or can plausibly predict about "
		// + "Apple's next MacBook Air laptop: a laptop that we're pretty sure
		// will come with a "
		// + "Retina display. We're also interested in rumours that Apple will
		// be launching its next "
		// + "MacBook Air in a new size: one based on a 12in screen.");

		/*
		 * String text = new FNCFeatureEngineering() .cleanText(
		 * "\"Let's get this vis-a-vis\", he said, \"these boys' marks are really that well?\""
		 * );
		 * 
		 * System.out.println(text);
		 * 
		 * System.out.println(getNGrams(text, 3));
		 * System.out.println(getCharGrams(text, 3));
		 */
		// System.out.println(new FNCFeatureEngineering().lemmatize(text));

		// System.out.println(removeStopWords(new
		// FNCFeatureEngineering().lemmatize(text)));

		pvecs = WordVectorSerializer.readParagraphVectors("resources/docvec_TEST_wFreq2");
		setData();
		setLabelsForDoc2VecFeatures();
		
		/*List<Double> titleBodyVec = new ArrayList<>();
		System.out.println(25386 + " : " + titleBodyPairs.get(25386).get(0));
		INDArray titleVec = pvecs.getLookupTable().vector(titleBodyPairs.get(25386).get(0)); // get the text vector by the id / label
		for(int k = 0; k < pvecs.getLayerSize(); k++){ //line 1383 in the example
			titleBodyVec.add(titleVec.getDouble(k)); //feature_k of body 0
		}
		
		INDArray bodyVec = pvecs.getLookupTable().vector(titleBodyPairs.get(25385).get(1)); // get the text vector by the id / label
		for(int k = 0; k < pvecs.getLayerSize(); k++){ //line 1383 in the example
			titleBodyVec.add(bodyVec.getDouble(k)); //feature_k of body 0
		}*/
		
		
		getWekaInstances();
		saveInstancesToARFFFile("resources/TEST_features_doc2vec_base_freq2_withvecs.arff");

	}

	private static void setLabelsForDoc2VecFeatures() throws FileNotFoundException, IOException {
		
		List<List<String>> stances = FNCUtility.readStances("resources/train_stances.csv");
		System.out.println(stances.size());
		stances.addAll(FNCUtility.readStances("resources/competition_test_stances.csv"));
		titleIdMap = new HashMap<>();
		List<List<String>> titleBodyPairs = new ArrayList<>();
		int i = 0;
		int j = 0; //stances
		for (List<String> stance : stances) {
			if (!titleIdMap.containsKey(stance.get(0))) {
				titleIdMap.put(stance.get(0),"title_" + i);

				List<String> pair = new ArrayList<>();
				pair.add("title_" + i);
				pair.add(stance.get(1));
				titleBodyPairs.add(pair);
				i++;
			
			} else {

				List<String> pair = new ArrayList<>();
				pair.add(titleIdMap.get(stance.get(0)));
				pair.add(stance.get(1));
				titleBodyPairs.add(pair);
			}
			j++;
			if(j==49972)
				System.out.println(i);
		}
		
		
		
		/*
		List<String> titlesAndBodiesList = new ArrayList<>();
		List<String> idsList = new ArrayList<>();
		for (Entry<Integer, String> e : idBodyMap.entrySet()) {
			titlesAndBodiesList.add(e.getValue());
			idsList.add(String.valueOf(e.getKey()));
		}

		titleIdMap = new HashMap<>();
		titleBodyPairs = new ArrayList<>();
		int i = 1648;
		for (List<String> stance : stances) {
			if (!titleIdMap.containsKey(stance.get(0))) {
				titleIdMap.put(stance.get(0), "title_" + i);
				titlesAndBodiesList.add(stance.get(0));
				idsList.add("title_" + i);

				List<String> pair = new ArrayList<>();
				pair.add("title_" + i);
				pair.add(stance.get(1));
				titleBodyPairs.add(pair);
				i++;
			} else {

				List<String> pair = new ArrayList<>();
				pair.add(titleIdMap.get(stance.get(0)));
				pair.add(stance.get(1));
				titleBodyPairs.add(pair);
			}
		}
		System.out.println("i = " + i);
		*/

	}

}

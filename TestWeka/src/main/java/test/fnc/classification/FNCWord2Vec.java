package test.fnc.classification;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.TreeSet;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class FNCWord2Vec {

	private static TreeSet<String> stopSet;
	
	private static HashMap<Integer, String> idBodyMap;

	static List<List<String>> stances;

	static Instances instances;

	private static WordVectors vec;

	public static void setData(String bodiesFilePath, String stancesFilePath) throws IOException {

		idBodyMap = FNCUtility.getBodiesMap(bodiesFilePath);

		stances = FNCUtility.readStances(stancesFilePath);
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

	public static void initializeStopwords(String stopFile) throws Exception {
		stopSet = new TreeSet<>();
		Scanner s = new Scanner(new FileReader(stopFile));
		while (s.hasNext())
			stopSet.add(s.next());
		s.close();
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
	private static Instances getWekaInstances() {
		ArrayList<Attribute> attributes = new ArrayList<>();
		String stancesClasses[] = new String[] { "agree", "disagree", "discuss", "unrelated" };
		List<String> stanceValues = Arrays.asList(stancesClasses);
		attributes.add(new Attribute("class", stanceValues));

		attributes.add(new Attribute("word2Vec_sim"));

		attributes.add(new Attribute("title", (ArrayList<String>) null));
		attributes.add(new Attribute("body_id"));

		instances = new Instances("fnc", attributes, 1000);
		for (int i = 0; i < 20; i++) {

			if ((i + 1) % 1000 == 0)
				System.out.println("Processing after i " + i);

			List<Double> features = new ArrayList<>();

			// class attribute
			features.add((double) stanceValues.indexOf(stances.get(i).get(2)));

			String headline = getLemmatizedInput(stances.get(i).get(0));

			System.out.println("here 4");
			List<String> headlineTokens = preprocess(headline);
			headlineTokens = removeStopWords(headlineTokens);
			// get the word2vec of the headline
			// To calculate similarity we may not just get the mean vector.. but
			// everz word vector
			// INDArray hMeanVec = vec.getWordVectors(headlineTokens);
			// System.out.println(hMeanVec.shapeInfoToString());
			// hMeanVec = hMeanVec.reshape(1, -1);
			// System.out.println(meanVec.getRow(0).getDouble(0));
			// System.out.println(meanVec.rows());
			// System.out.println(hMeanVec.shapeInfoToString());
			// System.out.println(Nd4j.toFlattened(meanVec));

			/*
			 * hW2VFeatures = for(int j = 0; j < 300; j++){
			 * meanVec.getRow(0).getDouble(j); }
			 */
			double[] vh = new double[300];
			INDArray ndv = Nd4j.create(vh);
			ndv = ndv.mul(0.0);
			int hn = 0;
			for (String htoken : headlineTokens) {
				if (vec.hasWord(htoken)) {
					INDArray hvec = vec.getWordVectorMatrix(htoken);
					ndv = ndv.add(hvec);
					System.out.print(htoken + " ");
					hn++;
				}
			}
			System.out.println();
			System.out.println(ndv);
			// ndv = Nd4j.norm2(ndv);
			// ndv = ndv.reshape(1, -1);
			ndv = ndv.div(hn);
			System.out.println(ndv);
			// features.add(instances.attribute(1).addStringValue(headline));
			System.out.println("body");
			String body = getLemmatizedInput(idBodyMap.get(Integer.valueOf(stances.get(i).get(1))));
			List<String> bodyTokens = preprocess(body);
			bodyTokens = removeStopWords(bodyTokens);

			double[] vb = new double[300];
			INDArray ndb = Nd4j.create(vb);
			ndb = ndb.mul(0.0);
			int bn = 0;
			for (String btoken : bodyTokens) {
				if (vec.hasWord(btoken)) {
					INDArray bvec = vec.getWordVectorMatrix(btoken);
					ndb = ndb.add(bvec);
					System.out.print(btoken + " ");
					bn++;
				}
			}
			System.out.println(ndb);
			// ndb = Nd4j.norm2(ndb);
			// ndb = ndb.reshape(1, -1);
			ndb = ndb.div(bn);
			System.out.println(ndb);

			// INDArray bMeanVec = vec.getWordVectors(bodyTokens);
			// System.out.println(bMeanVec.shapeInfoToString());
			// bMeanVec = bMeanVec.reshape(1, -1);
			// System.out.println(bMeanVec.shapeInfoToString());

			double cosineSim = Transforms.cosineSim(ndv, ndb);
			//features.add(cosineSim);
			
			System.out.println("dis1 = " + ndv.distance1(ndb));
			//System.out.println("dis2 = " + ndv.distance2(ndb));
			features.add(1 - ndv.distance2(ndb));
			//ndv.cos

			// adding title and body_id
			List<Double> values = new ArrayList<>();

			values.add((double) instances.attribute(2).addStringValue(headline));
			values.add((double) Integer.valueOf(stances.get(i).get(1)));

			features.addAll(values);

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

	private static List<String> preprocess(String headline) {
		TokenizerFactory t = new DefaultTokenizerFactory();

		/*
		 * CommonPreprocessor will apply the following regex to each token:
		 * [\d\.:,"'\(\)\[\]|/?!;]+ So, effectively all numbers, punctuation
		 * symbols and some special symbols are stripped off. Additionally it
		 * forces lower case for all tokens.
		 */
		t.setTokenPreProcessor(new CommonPreprocessor());
		List<String> tokens = t.create(headline).getTokens();

		/*
		 * String processed = ""; for(String token: tokens){ processed += token
		 * + " "; } processed = processed.trim();
		 */

		return tokens;
	}

	public static void main(String[] args) throws Exception {

		vec = WordVectorSerializer.readWord2VecModel(new File("resources/GoogleNews-vectors-negative300.bin.gz"));
		initializeStopwords("resources/stopwords.txt");
		
		System.out.println("here 1");
		setData("resources/train_bodies.csv", "resources/train_stances.csv");
		System.out.println("here 2");
		getWekaInstances();
		System.out.println("here 3");
		saveInstancesToARFFFile("resources/w2v_sim_features_add.arff");
	}

	private static void saveInstancesToARFFFile(String filepath) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new java.io.File(filepath));
		saver.writeBatch();
	}
}

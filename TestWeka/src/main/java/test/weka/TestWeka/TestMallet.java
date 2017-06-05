package test.weka.TestWeka;

import static com.google.common.base.Predicates.in;
import static org.simmetrics.builders.StringDistanceBuilder.with;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

import org.simmetrics.StringDistance;
import org.simmetrics.metrics.CosineSimilarity;
import org.simmetrics.metrics.EuclideanDistance;
import org.simmetrics.simplifiers.Simplifiers;
import org.simmetrics.tokenizers.Tokenizers;

import com.google.common.base.Predicates;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;
import weka.core.DenseInstance;

public class TestMallet {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			System.out.println("File not found" + filename);
		}
		return inputReader;

	}

	public static TreeSet<String> getStopwords(String file) throws FileNotFoundException {
		TreeSet<String> stopWords = new TreeSet<>();
		Scanner s = new Scanner(new FileReader(file));
		while (s.hasNext())
			stopWords.add(s.next());
		s.close();
		return stopWords;
	}

	public static float preprocess(String a, String b) throws FileNotFoundException {
		Set<String> commonWords = Sets.newHashSet(getStopwords("resources/stopwords.txt"));
		// System.out.println(commonWords);
		StringDistance metric = with(new CosineSimilarity<String>()).simplify(Simplifiers.toLowerCase())
				.simplify(Simplifiers.removeNonWord()).tokenize(Tokenizers.whitespace())
				.filter(Predicates.not(in(commonWords))).tokenize(Tokenizers.qGram(3)).build();

		// similarity(a,b) = 1 - distance(a,b) / √(∣a∣² + ∣b∣²)

		// float similarity = 1 - metric.distance(a, b)/Math.sqrt(a)

		Cache<String, String> stringCache = CacheBuilder.newBuilder().maximumSize(2).build();

		Cache<String, Multiset<String>> tokenCache = CacheBuilder.newBuilder().maximumSize(2).build();

		// StringDistance metric = with(new
		// CosineSimilarity<String>()).simplify(Simplifiers.toLowerCase())
		// .simplify(Simplifiers.removeNonWord()).cacheStrings(stringCache).tokenize(Tokenizers.qGram(3))
		// .cacheTokens(tokenCache).build();
		return 1 - metric.distance(a, b); // 4.6904
	}

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("resources/test_bin.arff");
		Instances data = source.getDataSet();

		// System.out.println(data.get(3).stringValue(0));
		// System.out.println(data.get(3).stringValue(1));
		
		DataSource headerBIDsource = new DataSource("resources/headers.arff");
		Instances headerBIDdata = headerBIDsource.getDataSet();


		int numInst = data.numInstances();
		int numAtt = data.numAttributes();

		ArrayList<Attribute> atts = new ArrayList<>();
		
		
		atts.add(new Attribute("header", (ArrayList<String>) null));
		atts.add(new Attribute("bodey_ID"));
		
		atts.add(new Attribute("cos_sim"));

		String stances[] = new String[] { "unrelated", "related" };
		List<String> stanceValues = Arrays.asList(stances);
		atts.add(new Attribute("class", stanceValues));
		Instances finalInstances = new Instances("final_relation", atts, 1000);
		// double values[] = new double[finalInstances.numAttributes()];

		for (int i = 0; i < numInst; i++) {
			double values[] = new double[finalInstances.numAttributes()];
			
			String title = data.get(i).stringValue(0);
			String body = data.get(i).stringValue(1);
			if (body.length() - title.length() >= 1000)
				body = data.get(i).stringValue(1).substring(0, data.get(i).stringValue(1).length() / 2);

			float sim = preprocess(title, body);
			// System.out.println(
			// "Sim( " + data.get(i).stringValue(0) + " ,,,, " +
			// data.get(i).stringValue(1) + " )= " + sim);
			values[0] = finalInstances.attribute(0).addStringValue(title);
			values[1] = headerBIDdata.get(i).value(1);
			
			values[2] = sim;
			values[3] = stanceValues.indexOf(data.get(i).stringValue(2));
			// System.out.println(data.get(i).stringValue(2));
			finalInstances.add(new DenseInstance(1.0, values));
		}

		// System.out.println(finalInstances);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(finalInstances);
		saver.setFile(new java.io.File("resources/dist_feature_all.arff"));
		saver.writeBatch();

	}

}

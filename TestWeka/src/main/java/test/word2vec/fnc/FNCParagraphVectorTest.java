package test.word2vec.fnc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang.time.StopWatch;
import org.apache.commons.lang3.tuple.MutablePair;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import com.opencsv.CSVReader;


public class FNCParagraphVectorTest {

	private static Set<String> uniqueTitles;
	private static Map<String, String> titleIdMap;
	private static Map<String, String> idTitleMap;
	private static HashMap<String, String> idBodyMap;
	private static Map<String, String> allTextData;
	private static ArrayList<MutablePair<String, String>> titleBodyLablesPairs;

	private static int maxBodyLen;
	private static String maxBodyText;
	
	public static void main(String[] args) throws IOException {

		// 1. Get <body,id> map
		File bodiesFile = new File("resources/train_bodies.csv");
		idBodyMap = getBodiesMap(bodiesFile);

		getPairsLables("resources/train_stances.csv");

		// CSVWriter wri

		writeToFile();

		idTitleMap = new HashMap<>();

		// because we have the opposite mapping
		idTitleMap = getIdTileMap(titleIdMap);
		
		allTextData = new HashMap<>();
		allTextData.putAll(idTitleMap);
		allTextData.putAll(idBodyMap);
		
		String paraVecMdlFile = "mandocs" + allTextData.size() + ".txt";
		
		setDataStatistic();
		
		System.out.println("Max Body length = " + maxBodyLen);
		System.out.println(maxBodyText);
		
		

		//Vector Learning-related Settings
		boolean learnParaVecs = true;   //if set to false, pre-trained model will be loaded
		int minWordFrequency = 3;
		int wordLearnIterations = 100;
		int epochs = 10; //no of training epochs     
		int layerSize = 10;  /*length of a word/paragraph vector*/
		double lr = 0.025; //0.025
/*
		//learn
		ParagraphVectors vec = null;
		StopWatch st = new StopWatch();
		if(learnParaVecs) {
			vec = learnParagraphVectors(allTextData, paraVecMdlFile, minWordFrequency, wordLearnIterations, epochs, layerSize, lr);
		} 
		
		double sim = vec.similarity("Title-1","Body-712");
		System.out.println("Title-1/Body-712 similarity: " + sim);
		printParagraphVector("Title-1",  vec);
		printParagraphVector("Body-712",  vec);
		
		
		sim = vec.similarity("Title-2","Body-158");
		System.out.println("Title-2/Body-158 similarity: " + sim);
		printParagraphVector("Title-2",  vec);
		printParagraphVector("Body-158",  vec);

		System.out.println("\nEnd Test");
		
*/
	}

	/**Print a paragraphVector  */
	private static void printParagraphVector(String docid, ParagraphVectors vec) {
		if(vec.hasWord(docid)) {
			double[] V_city = vec.getWordVector(docid);
			System.out.print("\nVector of " + docid + ": " );
			for(int i=0; i< V_city.length; i++) {
				System.out.print(V_city[i] + " ");
			}
			System.out.println();
		}
	}

	private static ParagraphVectors learnParagraphVectors(Map<String,String> docContentsMap, String serialize2file,
			int minWordFrequency, int wordLearnIterations, int epochs, int layerSize, double lr)  {
		LabelsSource source = new LabelsSource();
		// build a iterator for our dataset
		FNCLabelAwareIterator iterator = new FNCLabelAwareIterator.Builder()
		.build(docContentsMap);

		AbstractCache<VocabWord> cache = new AbstractCache<VocabWord>();
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		StopWatch sw = new StopWatch();

		ParagraphVectors vec = new ParagraphVectors.Builder()
		.minWordFrequency(minWordFrequency)
		.iterations(wordLearnIterations)
		.epochs(epochs)     
		.layerSize(layerSize)  /*length of a paragraph vector*/
		.learningRate(lr)
		.labelsSource(source)
		.windowSize(5)
		.iterate(iterator)
		.trainWordVectors(true)
		.vocabCache(cache)
		.tokenizerFactory(t)
		.sampling(0)
		.build();

		sw.start();
		vec.fit();
		sw.stop();

		System.out.println("Time taken to learn ParagraphVectors for documents is " + sw.getTime() + "ms"); 

		//Serialising
		if(serialize2file != null && !serialize2file.isEmpty()) {
			WordVectorSerializer.writeParagraphVectors(vec, serialize2file);
		}
		return vec;
	}

	private static Map<String, String> getIdTileMap(Map<String, String> titleIdMap) {
		Map<String, String> idTitleMap = new HashMap<>();
		
		for (Map.Entry<String, String> entry : titleIdMap.entrySet()) {
			idTitleMap.put(entry.getValue(), entry.getKey());
		}
		return idTitleMap;
	}

	private static void writeToFile() throws IOException {
		BufferedWriter out = new BufferedWriter(new FileWriter("titleBodyLablesPairs.txt"));
		for (int i = 0; i < titleBodyLablesPairs.size(); i++) {
			out.write(titleBodyLablesPairs.get(i).toString());
			out.newLine();
		}
		out.close();
	}

	private static void getPairsLables(String file) throws IOException {
		CSVReader stancesReader = new CSVReader(new FileReader(file));
		String[] stancesline;

		uniqueTitles = new HashSet<>();
		titleBodyLablesPairs = new ArrayList<>();
		titleIdMap = new HashMap<>();
		stancesReader.readNext();
		while ((stancesline = stancesReader.readNext()) != null) {
			MutablePair<String, String> pair = new MutablePair<String, String>();
			pair.setRight("Body-" + stancesline[1]);

			if (!uniqueTitles.contains(stancesline[0])) {

				uniqueTitles.add(stancesline[0]);
				String id = "Title-" + uniqueTitles.size();
				titleIdMap.put(stancesline[0], id);
				pair.setLeft(id);
			} else {

				pair.setLeft(titleIdMap.get(stancesline[0]));
			}
			titleBodyLablesPairs.add(pair);
		}

		stancesReader.close();
	}

	private static HashMap<String, String> getBodiesMap(File bodiesFile) {
		HashMap<String, String> bodyMap = new HashMap<>(100, 100);
		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(bodiesFile));
			String[] line;
			line = reader.readNext();
			while ((line = reader.readNext()) != null) {
				bodyMap.put("Body-" + line[0], line[1]);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return bodyMap;
	}
	
	private static void setDataStatistic(){
		maxBodyLen = 0;
		for (Map.Entry<String, String> entry : idBodyMap.entrySet()) {
			if(entry.getValue().length() > maxBodyLen){
				maxBodyLen = entry.getValue().length();
				maxBodyText = entry.getValue();
			}
		}
	}

}

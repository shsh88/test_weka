package test.fnc.doc2vec;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import edu.stanford.nlp.io.EncodingPrintWriter.out;
import test.fnc.classification.FNCUtility;

public class Doc2Vec {
	private TokenizerFactory tokenizerFactory;
	private ParagraphVectors vec;

	public ParagraphVectors buildParagraphVectors(List<String> tweetMessagesList, List<String> labelSourceList) {
		SentenceIterator iter = new CollectionSentenceIterator(tweetMessagesList);
		AbstractCache<VocabWord> cache = new AbstractCache<VocabWord>();

		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		LabelsSource source = new LabelsSource(labelSourceList);

		vec = new ParagraphVectors.Builder().minWordFrequency(2).iterations(10).epochs(10).layerSize(100)
				.learningRate(0.025)
				// .minLearningRate(0.001)
				.labelsSource(source)
				// .stopWords(Files.readAllLines(new
				// File("../stopwords.txt").toPath(), Charset.defaultCharset()
				// ))
				.windowSize(10).iterate(iter).trainWordVectors(true).vocabCache(cache)
				// Wahlweise Distributional-BOW(default) oder Distributional
				// Memory new DM<VocabWord>()
				.sequenceLearningAlgorithm(new DM<VocabWord>()).tokenizerFactory(tokenizerFactory)
				// .sampling(0)
				.build();

		vec.fit();

		return vec;
	}

	public INDArray buildVectorFromUntrainedData(String message) {

		DocumentVectorBuilder meansBuilder = new DocumentVectorBuilder(
				(InMemoryLookupTable<VocabWord>) vec.getLookupTable(), tokenizerFactory);
		INDArray messageAsCentroid = meansBuilder.messageAsVector(message);

		return messageAsCentroid;

	}

	public static void main(String[] args) throws Exception {

		HashMap<Integer, String> idBodyMap = FNCUtility.getBodiesMap("resources/train_bodies.csv");
		idBodyMap.putAll(FNCUtility.getBodiesMap("resources/test_bodies.csv"));
		List<String> titlesBodiesList = new ArrayList<>();
		List<String> idsList = new ArrayList<>();
		for (Entry<Integer, String> e : idBodyMap.entrySet()) {
			titlesBodiesList.add(e.getValue());
			idsList.add(String.valueOf(e.getKey()));
		}

		List<List<String>> stances = FNCUtility.readStances("resources/train_stances.csv");
		System.out.println(stances.size());
		stances.addAll(FNCUtility.readStances("resources/competition_test_stances.csv"));
		Map<String, String> titleIdMap = new HashMap<>();
		List<List<String>> titleBodyPairs = new ArrayList<>();
		int i = 0;
		int j = 0; //stances
		for (List<String> stance : stances) {
			if (!titleIdMap.containsKey(stance.get(0))) {
				titleIdMap.put(stance.get(0),"title_" + i);
				titlesBodiesList.add(stance.get(0));
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
			j++;
			if(j==49972)
				System.out.println(i);
		}
		System.out.println("i = "+i);
		System.out.println("j = "+j);


		 Doc2Vec paraVec = new Doc2Vec();
		 ParagraphVectors docVec = paraVec.buildParagraphVectors(titlesBodiesList,
		 idsList);
		 WordVectorSerializer.writeParagraphVectors(docVec, "resources/docvec_TEST_wFreq2");

		ParagraphVectors pvecs = WordVectorSerializer.readParagraphVectors("resources/docvec_TEST_wFreq2");
		System.out.println(pvecs.similarWordsInVocabTo("isis", 0.9));
		// System.out.println(pvecs.similarityToLabel("hundred of Palestinians
		// flee flood in Gaza as Israel open dam", "158"));
		//int j = 0;
		/*for (List<String> p : titleBodyPairs) {
			System.out.println(p.get(0) + " : " + titleIdMap.get(p.get(0)) + " "+ p.get(1));
			System.out.println(pvecs.similarity(p.get(0), p.get(1)));
			j++;
			if (j == 20)
				break;
		}*/
		
		/*j = 0;
		for (List<String> stance : stances) {
			System.out.println(stance.get(0) + " : " + titleIdMap.get(stance.get(0)) + " "+ stance.get(1));
			System.out.println(pvecs.similarity(titleIdMap.get(stance.get(0)), stance.get(1)));
			
			j++;
			if (j == 20)
				break;
		}*/
		
		/*INDArray bodyVec = pvecs.getLookupTable().vector(titleBodyPairs.get(0).get(1)); // get the text vector by the id / label
		for(int k = 0; k < pvecs.getLayerSize(); k++){ //line 1383 in the example
			bodyVec.getDouble(k); //feature_k of body 0
		}*/
		
		INDArray errVec = pvecs.getLookupTable().vector("title_2537");
		for(int k = 0; k < pvecs.getLayerSize(); k++){ //line 1383 in the example
			System.out.println(errVec.getDouble(k)); //feature_k of body 0
		}

	}

}

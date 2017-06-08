package test.weka.TestWeka;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.time.StopWatch;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is example code for dl4j ParagraphVectors implementation. In this example we build distributed representation of all sentences present in training corpus.
 * However, you still use it for training on labelled documents, using sets of LabelledDocument and LabelAwareIterator implementation.
 *
 * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.8 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorLearningTest {

	private static final Logger log = LoggerFactory.getLogger(ParagraphVectorLearningTest.class);

	public static void main(String[] args) throws Exception {


		//        Manually create few test documents    	
		Map<String,String> docContentsMap = new HashMap<String,String>();
		docContentsMap.put("MYDOC_1","An article using the Redirect template may be used to place links within navigational elements where they would otherwise be unsupported or require different text than the standard. Maturity: Production/Stable ");
		docContentsMap.put("MYDOC_2","An article using the Redirect template may be used to place links within navigational elements where they would otherwise be unsupported or require different text than the standard. Maturity: Production/Stable ");
		docContentsMap.put("MYDOC_3","Woman detained in Lebanon is not al-Baghdadi's wife, Iraq says");
		//docContentsMap.put("MYDOC_3","We have compiled a list of frequently asked questions from residents and made them available online. Enter your question below and click search. If you don't find the answer to your question you can submit it for us to answer.");
		docContentsMap.put("MYDOC_4","An Iraqi official denied that a woman detained in Lebanon is a wife of Abu Bakr al-Baghdadi, the leader of the Islamic State group, adding that she is the sister of a terror suspect being held in Iraq.\n" + 
				"\n" + 
				"Wednesday's denial comes a day after Lebanese authorities said they are holding a woman believed to be al-Baghdadi's wife.\n" + 
				"\n" + 
				"They said she was detained for travelling with a fake ID and had herself claimed that she is the reclusive extremist leader's spouse.\n" + 
				"\n" + 
				"This file image made from video posted on a militant website purports to show the leader of the Islamic State group, Abu Bakr al-Baghdadi, delivering a sermon at a mosque in Iraq\n" + 
				"\n" + 
				"The statement by Iraq's Interior Ministry spokesman Saad Maan Ibrahim adds to the confusion surrounding the identity of the woman and child who were detained about 10 days ago in northern Lebanon travelling with a fake ID.\n" + 
				"\n" + 
				"Lebanese officials said the woman, Saja al-Dulaimi, is believed to be the wife of the reclusive IS leader. She was held by Syrian authorities and freed in a prisoner exchange with the Nusra Front, Syria's al-Qaida branch, earlier this year.\n" + 
				"\n" + 
				"The interrogation of the woman was being supervised by Lebanon's military prosecutor.\n" + 
				"\n" + 
				"It was unclear what would have brought the woman and child to Lebanon, where IS controls no territory and enjoys only small support in some predominantly Sunni Muslim areas.\n" + 
				"\n" + 
				"On Wednesday, Ibrahim told The Associated Press that al-Dulaimi, an Iraqi national who traveled to Syria before arriving in Lebanon, is not al-Baghdadi's wife. He identified her as the sister of Omar Abdul Hamid al-Dulaimi, who is being held in Iraq as a terror suspect.\n" + 
				"\n" + 
				"He added that al-Baghdadi has two wives but neither is named Saja al-Dulaimi. There was no immediate comment from Lebanese authorities.\n" + 
				"\n" + 
				"In Lebanon, a military expert was killed and two others wounded Wednesday when a bomb they were about to dismantle near the border with Syria exploded, the army said.\n" + 
				"\n" + 
				"The explosion comes a day after an ambush by suspected Islamic militants in the same area killed six soldiers and wounded one.\n" + 
				"\n" + 
				"In Lebanon, a military expert was killed and two others wounded Wednesday when a bomb they were about to dismantle near the border with Syria exploded. Lebanese soldiers are pictured here patrolling the area\n" + 
				"\n" + 
				"Lebanese troops (pictured) have been battling Syria-based Islamic militants, including the extremist Islamic State group and the al-Qaida-linked Nusra Front, in areas near the border\n" + 
				"\n" + 
				"No one has so far claimed responsibility for Tuesday's ambush or for planting the bomb that was discovered Wednesday.\n" + 
				"\n" + 
				"Lebanese troops have been battling Syria-based Islamic militants, including the extremist Islamic State group and the al-Qaida-linked Nusra Front, in areas near the border.\n" + 
				"\n" + 
				"Meanwhile, Saudi Arabia's Interior Ministry spokesman said police have not ruled out the possibility that Islamic State group supporters were behind the shooting of a Danish man last month.\n" + 
				"\n" + 
				"It was the first time a Saudi official publicly comments on the incident since a video was released by alleged IS supporters claiming responsibility for the drive-by shooting in Riyadh Nov. 22. The Danish citizen survived the shooting.\n" + 
				"\n" + 
				"Interior Ministry spokesman Mansour al-Turki's comments were published Wednesday in the Saudi Al-Eqtisadia newspaper.\n" + 
				"\n" + 
				"The video was released online this week by a group purporting to be IS supporters. It shows a gunman pulling up beside a vehicle and firing at the driver. It identifies the target as Thomas Hoepner. It was not immediately possible to confirm the authenticity of the video.\n" + 
				"\n" + 
				"Lebanese army special forces in armored personnel carriers patrol near the area militants ambushed Lebanese soldiers\n"); 
		//docContentsMap.put("MYDOC_4","Some general tips and tricks. Check the  for general tips and tricks when creating an article. Additional Downloads");
		String paraVecMdlFile = "mandocs" + docContentsMap.size() + ".txt";

		//Vector Learning-related Settings
		boolean learnParaVecs = true;   //if set to false, pre-trained model will be loaded
		int minWordFrequency = 1;
		int wordLearnIterations = 100;
		int epochs = 9; //no of training epochs     
		int layerSize = 10;  /*length of a word/paragraph vector*/
		double lr = 0.025; //0.025

		//learn
		ParagraphVectors vec = null;
		StopWatch st = new StopWatch();
		if(learnParaVecs) {
			vec = learnParagraphVectors(docContentsMap, paraVecMdlFile, minWordFrequency, wordLearnIterations, epochs, layerSize, lr);
		} /* else {
			st.reset();
			st.start();
			vec =  WordVectorSerializer.readParagraphVectorsFromText(paraVecMdlFile);
			st.stop();
			System.out.println("Time taken for reading paragraphVectors from disk: " + st.getTime() + "ms");
		}*/

		double sim = vec.similarity("MYDOC_1", "MYDOC_2");
		log.info("MYDOC_1/MYDOC_2 similarity: " + sim);
		System.out.println("MYDOC_1/MYDOC_2 similarity: " + sim);
		printParagraphVector("MYDOC_3",  vec);
		printParagraphVector("MYDOC_4",  vec);

		System.out.println("\nEnd Test");
	} //end main()



	//==================Utility methods==============    
	private static ParagraphVectors learnParagraphVectors(Map<String,String> docContentsMap, String serialize2file,
			int minWordFrequency, int wordLearnIterations, int epochs, int layerSize, double lr) throws IOException {

		LabelsSource source = new LabelsSource();
		// build a iterator for our dataset
		SolrDocLabelAwareIterator2 iterator = new SolrDocLabelAwareIterator2.Builder()
		.build(docContentsMap);

		InMemoryLookupCache cache = new InMemoryLookupCache();
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		StopWatch sw = new StopWatch();

		ParagraphVectors vec = new ParagraphVectors.Builder()
		.minWordFrequency(minWordFrequency)
		.iterations(wordLearnIterations)
		.epochs(epochs)     
		.layerSize(layerSize)  /*length of a paragraph vector*/
		.learningRate(lr)
		//.labelsSource(source)
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
			WordVectorSerializer.writeWordVectors(vec, serialize2file);
		}
		return vec;
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


}

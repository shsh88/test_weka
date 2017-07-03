package test.word2vec.fnc;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import cc.mallet.pipe.tsf.WordVectors;

public class TestWithGoogleCurpus {
	public static void main(String[] args) throws IOException {
		//File gModel = new File("resources/GoogleNews-vectors-negative300.bin.gz");

		Word2Vec vec = WordVectorSerializer.readWord2VecModel("resources/GoogleNews-vectors-negative300.bin.gz", false);

		InputStreamReader r = new InputStreamReader(System.in);

		BufferedReader br = new BufferedReader(r);

		for (;;) {
			System.out.print("Word: ");
			String word = br.readLine();

			if ("EXIT".equals(word))
				break;

			Collection<String> lst = vec.wordsNearest(word, 20);

			System.out.println(word + " -> " + lst);
			
			//WordVectors wv = new WordVectors(prefix, vectorsFile)
		}
	}
}

package test.weka.TestWeka;

/*
 * #%L
 * Simmetrics Examples
 * %%
 * Copyright (C) 2014 - 2016 Simmetrics Authors
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */

import static org.simmetrics.builders.StringDistanceBuilder.with;

import org.simmetrics.StringDistance;
import org.simmetrics.metrics.EuclideanDistance;
import org.simmetrics.metrics.StringDistances;
import org.simmetrics.tokenizers.Tokenizers;

/**
 * The StringDistances utility class contains a predefined list of well known
 * distance metrics for strings.
 */
public final class StringDistanceExample {

	/**
	 * Two strings can be compared using a predefined distance metric.
	 */
	public static float example01() {

		String str1 = "Police find mass graves with at least '15 bodies' near Mexico town where 43 students disappeared after police clash";
		String str2 = "Danny Boyle is directing the untitled film\n" + 
				"\n" + 
				"Seth Rogen is being eyed to play Apple co-founder Steve Wozniak in Sony’s Steve Jobs biopic.\n" + 
				"\n" + 
				"Danny Boyle is directing the untitled film, based on Walter Isaacson's book and adapted by Aaron Sorkin, which is one of the most anticipated biopics in recent years.\n" + 
				"\n" + 
				"Negotiations have not yet begun, and it’s not even clear if Rogen has an official offer, but the producers — Scott Rudin, Guymon Casady and Mark Gordon — have set their sights on the talent and are in talks.\n" + 
				"\n" + 
				"Of course, this may all be for naught as Christian Bale, the actor who is to play Jobs, is still in the midst of closing his deal. Sources say that dealmaking process is in a sensitive stage.\n" + 
				"\n" + 
				"Insiders say Boyle will is flying to Los Angeles to meet with actress to play one of the female leads, an assistant to Jobs. Insiders say that Jessica Chastain is one of the actresses on the meeting list.\n" + 
				"\n" + 
				"Wozniak, known as \"Woz,\" co-founded Apple with Jobs and Ronald Wayne. He first met Jobs when they worked at Atari and later was responsible for creating the early Apple computers.\n" + 
				"";

		StringDistance metric = StringDistances.levenshtein();

		return metric.distance(str1, str2); // 30.0000
	}

	/**
	 * A tokenizer is included when the metric is a set or list metric. For the
	 * euclidean distance, it is a whitespace tokenizer.
	 * 
	 * Note that most predefined metrics are setup with a whitespace tokenizer.
	 */
	public static float example02() {

		//String str1 = "A quirky thing it is. This is a sentence.";
		//String str2 = "This sentence is similar. A quirky thing it is.";
		String str1 = "Police find mass graves with at least '15 bodies' near Mexico town where 43 students disappeared after police clash";
		String str2 = "Danny Boyle is directing the untitled film\n" + 
				"\n" + 
				"Seth Rogen is being eyed to play Apple co-founder Steve Wozniak in Sony’s Steve Jobs biopic.\n" + 
				"\n" + 
				"Danny Boyle is directing the untitled film, based on Walter Isaacson's book and adapted by Aaron Sorkin, which is one of the most anticipated biopics in recent years.\n" + 
				"\n" + 
				"Negotiations have not yet begun, and it’s not even clear if Rogen has an official offer, but the producers — Scott Rudin, Guymon Casady and Mark Gordon — have set their sights on the talent and are in talks.\n" + 
				"\n" + 
				"Of course, this may all be for naught as Christian Bale, the actor who is to play Jobs, is still in the midst of closing his deal. Sources say that dealmaking process is in a sensitive stage.\n" + 
				"\n" + 
				"Insiders say Boyle will is flying to Los Angeles to meet with actress to play one of the female leads, an assistant to Jobs. Insiders say that Jessica Chastain is one of the actresses on the meeting list.\n" + 
				"\n" + 
				"Wozniak, known as \"Woz,\" co-founded Apple with Jobs and Ronald Wayne. He first met Jobs when they worked at Atari and later was responsible for creating the early Apple computers.\n" + 
				"";

		StringDistance metric = StringDistances.euclideanDistance();

		return metric.distance(str1, str2); // 2.0000
	}

	/**
	 * Using the string distance builder distance metrics can be customized.
	 * Instead of a whitespace tokenizer a q-gram tokenizer is used.
	 *
	 * For more examples see StringDistanceBuilderExample.
	 */
	public static float example03() {

		//String str1 = "A quirky thing it is. This is a sentence.";
		//String str2 = "This sentence is similar. A quirky thing it is.";
		
		String str1 = "Police find mass graves with at least '15 bodies' near Mexico town where 43 students disappeared after police clash";
		String str2 = "Danny Boyle is directing the untitled film\n" + 
				"\n" + 
				"Seth Rogen is being eyed to play Apple co-founder Steve Wozniak in Sony’s Steve Jobs biopic.\n" + 
				"\n" + 
				"Danny Boyle is directing the untitled film, based on Walter Isaacson's book and adapted by Aaron Sorkin, which is one of the most anticipated biopics in recent years.\n" + 
				"\n" + 
				"Negotiations have not yet begun, and it’s not even clear if Rogen has an official offer, but the producers — Scott Rudin, Guymon Casady and Mark Gordon — have set their sights on the talent and are in talks.\n" + 
				"\n" + 
				"Of course, this may all be for naught as Christian Bale, the actor who is to play Jobs, is still in the midst of closing his deal. Sources say that dealmaking process is in a sensitive stage.\n" + 
				"\n" + 
				"Insiders say Boyle will is flying to Los Angeles to meet with actress to play one of the female leads, an assistant to Jobs. Insiders say that Jessica Chastain is one of the actresses on the meeting list.\n" + 
				"\n" + 
				"Wozniak, known as \"Woz,\" co-founded Apple with Jobs and Ronald Wayne. He first met Jobs when they worked at Atari and later was responsible for creating the early Apple computers.\n" + 
				"";

		StringDistance metric = with(new EuclideanDistance<String>()).tokenize(Tokenizers.qGram(3)).build();

		return metric.distance(str1, str2); // 4.8989
	}

	public static void main(String[] args) {
		System.out.println(example01());
		System.out.println(example02());
		System.out.println(example03());
	}

}

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

import static com.google.common.base.Predicates.in;
import static org.simmetrics.builders.StringDistanceBuilder.with;

import java.util.Set;

import org.simmetrics.StringDistance;
import org.simmetrics.metrics.CosineSimilarity;
import org.simmetrics.metrics.EuclideanDistance;
import org.simmetrics.metrics.Levenshtein;
import org.simmetrics.simplifiers.Simplifiers;
import org.simmetrics.tokenizers.Tokenizers;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;

/**
 * The string distance builder can be used to compose distance metrics for
 * strings.
 */
public final class StringDistanceBuilderExample {

	/**
	 * Simply comparing strings through a metric may not be very effective. By
	 * adding simplifiers, tokenizers and filters and transform the
	 * effectiveness of a metric can be improved.
	 * 
	 * The exact combination is generally domain specific. The
	 * StringDistanceBuilder supports these domain specific customizations. Some
	 * example usages are shown below
	 */
	public static float example00() {

		String a = "Chilpéric II son of Childeric II";
		String b = "chilperic ii son of childeric ii";

		StringDistance metric = new Levenshtein();

		return metric.distance(a, b); // 7.0000
	}

	/**
	 * Simplification
	 * 
	 * Simplification increases the effectiveness of a metric by removing noise
	 * and reducing the dimensionality of the problem. The process maps a a
	 * complex string to a simpler format. This allows string from different
	 * sources to be compared in the same form.
	 *
	 * The Simplifiers utility class contains a collection of common, useful
	 * simplifiers. For a custom simplifier you can implement the Simplifier
	 * interface.
	 */
	public static float example01() {

		String a = "Chilpéric II son of Childeric II";
		String b = "Chilperic II son of Childeric II";

		StringDistance metric = with(new Levenshtein()).simplify(Simplifiers.removeDiacritics()).build();

		return metric.distance(a, b); // 0.0000
	}

	/**
	 * Simplifiers can also be chained.
	 */
	public static float example02() {

		String a = "Chilpéric II son of Childeric II";
		String b = "chilperic ii son of childeric ii";

		StringDistance metric = with(new Levenshtein()).simplify(Simplifiers.removeDiacritics())
				.simplify(Simplifiers.toLowerCase()).build();

		return metric.distance(a, b); // 0.0000
	}

	/**
	 * Tokenization
	 * 
	 * A metric can be used to measure the distance between strings. However not
	 * all metrics can operate on strings directly. Some operate on lists, sets
	 * or multisets. To compare strings with a metric that works on a collection
	 * a tokenizer is required. Tokenization cuts up a string into parts.
	 * 
	 * Example:
	 * 
	 * `chilperic ii son of childeric ii`
	 * 
	 * By splitting on whitespace is tokenized into:
	 * 
	 * `[chilperic, ii, son, of, childeric, ii]`
	 * 
	 * The choice of the tokenizer can influence the effectiveness of a metric.
	 * For example when comparing individual words a q-gram tokenizer will be
	 * more effective while a whitespace tokenizer will be more effective when
	 * comparing documents.
	 * 
	 * The Tokenizers utility class contains a collection of common, useful
	 * tokenizers. For a custom tokenizer you can implement the Tokenizer
	 * interface. Though it is recommended that you extend the
	 * AbstractTokenizer.
	 */
	public static float example03() {

		String a = "A quirky thing it is. This is a sentence.";
		String b = "This sentence is similar; a quirky thing it is.";

		StringDistance metric = with(new EuclideanDistance<String>()).tokenize(Tokenizers.whitespace()).build();

		return metric.distance(a, b); // 2.0000
	}

	/**
	 * Tokenizers can also be chained.
	 * 
	 * `chilperic ii son of childeric ii`
	 * 
	 * By splitting on whitespace is tokenized into:
	 * 
	 * `[chilperic, ii, son, of, childeric, ii]`
	 * 
	 * After using a q-gram with a q of 2:
	 * 
	 * `[ch,hi,il,il,lp,pe,er,ri,ic, ii, so,on, of, ch,hi,il,ld,de,er,ri,ic,
	 * ii]`
	 * 
	 */
	public static float example04() {

		String a = "A quirky thing it is. This is a sentence.";
		String b = "This sentence is similar; a quirky thing it is.";

		StringDistance metric = with(new EuclideanDistance<String>()).tokenize(Tokenizers.whitespace())
				.tokenize(Tokenizers.qGram(3)).build();

		return metric.distance(a, b); // 2.8284
	}

	/**
	 * Tokens can be filtered to avoid comparing strings on common but otherwise
	 * low information words. Tokens can be filtered after any tokenization step
	 * and filters can be applied repeatedly.
	 * 
	 * A filter can be implemented by implementing a the {@link Predicate}
	 * interface. By chaining predicates more complicated filters can be build.
	 */
	public static float example05() {
		Set<String> commonWords = Sets.newHashSet("it", "is");
		Set<String> otherCommonWords = Sets.newHashSet("a");

		String a = "A quirky thing it is. This is a sentence.";
		String b = "This sentence is similar; a quirky thing it is.";

		StringDistance metric = with(new EuclideanDistance<String>()).simplify(Simplifiers.toLowerCase())
				.simplify(Simplifiers.removeNonWord()).tokenize(Tokenizers.whitespace())
				.filter(Predicates.not(in(commonWords))).filter(Predicates.not(in(otherCommonWords)))
				.tokenize(Tokenizers.qGram(3)).build();

		return metric.distance(a, b); // 4.6904
	}

	/**
	 * Tokens can be transformed to a simpler form. This may be used to reduce
	 * the possible token space. Tokens can be transformed after any
	 * tokenization step and the transformation can be applied repeatedly.
	 * 
	 * A transformation can be implemented by implementing a the Function
	 * interface.
	 */
	public static float example06() {

		Function<String, String> reverse = new Function<String, String>() {

			@Override
			public String apply(String input) {
				return new StringBuilder(input).reverse().toString();
			}

		};

		String a = "A quirky thing it is. This is a sentence.";
		String b = "This sentence is similar; a quirky thing it is.";

		StringDistance metric = with(new EuclideanDistance<String>()).simplify(Simplifiers.toLowerCase())
				.simplify(Simplifiers.removeNonWord()).tokenize(Tokenizers.whitespace()).transform(reverse)
				.tokenize(Tokenizers.qGram(3)).build();

		return metric.distance(a, b); // 4.6904
	}

	/**
	 * Tokenization and simplification can be expensive operations. To avoid
	 * executing expensive operations repeatedly, intermediate results can be
	 * cached. Note that Caching itself also has a non-trivial cost. Base your
	 * decision on metrics!
	 */
	public static float example07() {

		//String a = "Police find mass graves with at least '15 bodies' near Mexico town where 43 students disappeared after police clash";
		//String b = "Danny Boyle is directing the untitled film Seth Rogen is being eyed to play Apple co-founder Steve Wozniak in Sony’s Steve Jobs biopic.\\n\\nDanny Boyle is directing the untitled film, based on Walter Isaacson\\'s book and adapted by Aaron Sorkin, which is one of the most anticipated biopics in recent years. Negotiations have not yet begun, and it’s not even clear if Rogen has an official offer, but the producers — Scott Rudin, Guymon Casady and Mark Gordon — have set their sights on the talent and are in talks.\\n\\nOf course, this may all be for naught as Christian Bale, the actor who is to play Jobs, is still in the midst of closing his deal. Sources say that dealmaking process is in a sensitive stage.\\n\\nInsiders say Boyle will is flying to Los Angeles to meet with actress to play one of the female leads, an assistant to Jobs. Insiders say that Jessica Chastain is one of the actresses on the meeting list.nWozniak, known as\"Woz,\" co-founded Apple with Jobs and Ronald Wayne. He first met Jobs when they worked at Atari and later was responsible for creating the early Apple computers.";

		String a = "A quirky thing it is. This is a sentence.";
		String b = "This sentence is similar; a quirky thing it is.";
		
		Cache<String, String> stringCache = CacheBuilder.newBuilder().maximumSize(2).build();

		Cache<String, Multiset<String>> tokenCache = CacheBuilder.newBuilder().maximumSize(2).build();

		StringDistance metric = with(new CosineSimilarity<String>()).simplify(Simplifiers.toLowerCase())
				.simplify(Simplifiers.removeNonWord()).cacheStrings(stringCache).tokenize(Tokenizers.qGram(3))
				.cacheTokens(tokenCache).build();

		return metric.distance(a, b); // 4.6904
	}

	public static void main(String[] args) {
		System.out.println(example07());

	}

}

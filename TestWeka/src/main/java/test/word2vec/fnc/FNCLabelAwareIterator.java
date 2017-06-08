package test.word2vec.fnc;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;

import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;

public class FNCLabelAwareIterator implements LabelAwareIterator {
	private Iterator<String> iter;
	private Map<String, String> docContentsMap;
	protected LabelsSource labelsSource;

	public FNCLabelAwareIterator(Map<String, String> doccontentsmap, @Nonnull LabelsSource source) {
		docContentsMap = doccontentsmap;
		iter = docContentsMap.keySet().iterator();
		labelsSource = source;
	}

	@Override
	public boolean hasNext() {
		return iter.hasNext();
	}

	@Override
	public LabelledDocument next() {
		LabelledDocument document = new LabelledDocument();
		if (iter.hasNext()) {
			String label = iter.next();
			String txt = docContentsMap.get(label);
			if (txt.length() < 30) {
				txt = " insufficient content to cluster this document";
			}
			document.setContent(txt);
			document.addLabel(label);
		}
		return document;
	}

	@Override
	public boolean hasNextDocument() {
		return iter.hasNext();
	}

	@Override
	public LabelledDocument nextDocument() {
		LabelledDocument document = new LabelledDocument();
		if (iter.hasNext()) {
			String label = iter.next();
			String txt = docContentsMap.get(label);
			if (txt.length() < 30) {
				txt = " insufficient content to cluster this document";
			}
			document.setContent(txt);
			document.setLabel(label);
		}
		return document;
	}

	@Override
	public void reset() {
		iter = docContentsMap.keySet().iterator();

	}

	@Override
	public LabelsSource getLabelsSource() {
		return labelsSource;
	}

	@Override
	public void shutdown() {
		// TODO Auto-generated method stub

	}

	public static class Builder {
		public Builder() {
		}

		public FNCLabelAwareIterator build(Map<String, String> doccontentsmap) {
			List<String> labels = new ArrayList<>();
			for (String docid : doccontentsmap.keySet()) {
				labels.add(docid);
			}
			LabelsSource source = new LabelsSource(labels);
			FNCLabelAwareIterator iterator = new FNCLabelAwareIterator(doccontentsmap, source);

			return iterator;
		}
	}

}

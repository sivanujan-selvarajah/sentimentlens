package ch.zhaw.sentimentlens.translate;

import ai.djl.translate.*;
import ai.djl.ndarray.*;
import ai.djl.modality.Classifications;

import java.util.*;

public class SentimentTranslator2 implements Translator<String, Classifications> {

    private final List<String> vocabulary;

    public SentimentTranslator2(List<String> vocabulary) {
        this.vocabulary = vocabulary;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        NDManager manager = ctx.getNDManager();
        String[] tokens = input.toLowerCase().split("\\W+");

        // TF: Häufigkeit jedes Tokens im aktuellen Dokument
        Map<String, Integer> tokenCounts = new HashMap<>();
        for (String token : tokens) {
            tokenCounts.put(token, tokenCounts.getOrDefault(token, 0) + 1);
        }

        int totalTerms = tokens.length;
        float[] tfidfVector = new float[vocabulary.size()];

        for (int i = 0; i < vocabulary.size(); i++) {
            String word = vocabulary.get(i);
            int tf = tokenCounts.getOrDefault(word, 0);

            // Vereinfachte IDF-Schätzung: log(1 + N / (1 + df)) → hier statisch, da df unbekannt
            float idf = (float) Math.log(1 + (1.0 / (1 + tf))); // "inverse document frequency" Schätzung

            tfidfVector[i] = (tf / (float) totalTerms) * idf;
        }

        NDArray array = manager.create(tfidfVector);
        return new NDList(array);
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray rawOutput = list.singletonOrThrow();            // Logits
        NDArray softmaxProbs = rawOutput.softmax(0);            // Softmax-Wahrscheinlichkeiten

        System.out.println("🔍 Wahrscheinlichkeiten: " + softmaxProbs);

        List<String> classes = List.of("positiv", "negativ");
        return new Classifications(classes, softmaxProbs);
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
}

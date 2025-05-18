/*  package ch.zhaw.sentimentlens.translate;

import ai.djl.translate.*;
import ai.djl.ndarray.*;
import ai.djl.modality.Classifications;

import java.util.*;

public class SentimentTranslator implements Translator<String, Classifications> {

    private final List<String> vocabulary;

    public SentimentTranslator(List<String> vocabulary) {
        this.vocabulary = vocabulary;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        NDManager manager = ctx.getNDManager();
        String[] tokens = input.toLowerCase().split("\\W+");

        Map<String, Integer> tokenCounts = new HashMap<>();
        for (String token : tokens) {
            tokenCounts.put(token, tokenCounts.getOrDefault(token, 0) + 1);
        }

        float[] vector = new float[vocabulary.size()];
        for (int i = 0; i < vocabulary.size(); i++) {
            vector[i] = tokenCounts.getOrDefault(vocabulary.get(i), 0);
        }

        NDArray array = manager.create(vector);
        return new NDList(array);
    }

    @Override
public Classifications processOutput(TranslatorContext ctx, NDList list) {
    NDArray rawOutput = list.singletonOrThrow();            // logits = rohes Modell-Output
    NDArray softmaxProbs = rawOutput.softmax(0);            // softmax auf das Modell-Output

    System.out.println("🔍 Wahrscheinlichkeiten: " + softmaxProbs);

    List<String> classes = List.of("positiv", "negativ");
    return new Classifications(classes, softmaxProbs);
}

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }
} */

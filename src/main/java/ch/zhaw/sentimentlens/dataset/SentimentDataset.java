package ch.zhaw.sentimentlens.dataset;


import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.modality.nlp.preprocess.Tokenizer;
import ai.djl.translate.TranslateException;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class SentimentDataset {

    private List<String> texts;
    private List<Integer> labels;

    public SentimentDataset() {
        texts = new ArrayList<>();
        labels = new ArrayList<>();
    }

    public void load(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("__label__1")) {
                    labels.add(0); // 0 = negativ
                    texts.add(line.replaceFirst("__label__1", "").trim());
                } else if (line.startsWith("__label__2")) {
                    labels.add(1); // 1 = positiv
                    texts.add(line.replaceFirst("__label__2", "").trim());
                }
            }
        }
        System.out.println("Dataset geladen: " + texts.size() + " Beispiele");
    }

    public List<String> getTexts() {
        return texts;
    }

    public List<Integer> getLabels() {
        return labels;
    }
}
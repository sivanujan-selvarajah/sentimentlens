/*  package ch.zhaw.sentimentlens.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Batchifier;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

@Service
public class SentimentService5 {

    private Model model;
    private Predictor<String, Classifications> predictor;

    @PostConstruct
    public void init() throws IOException, MalformedModelException {
        // Achtung: KEIN Name beim newInstance() – Spring Boot erwartet Default-Model
        model = Model.newInstance("pytorch");
        model.load(Paths.get("src/main/resources/model")); // Lade aus dem Ressourcenordner

        predictor = model.newPredictor(new SentimentTranslator());
    }

    public String analyzeSentiment(String text) throws TranslateException {
        Classifications classifications = predictor.predict(text);
        Classification best = classifications.best();
        return String.format("Erkanntes Sentiment: %s (%.2f%%)", best.getClassName(), best.getProbability() * 100);
    }

    // Innerer Translator
    private static class SentimentTranslator implements Translator<String, Classifications> {

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            NDManager manager = ctx.getNDManager();
            float feature = input.length() / 5.0f; // Gleiches Feature wie beim Training!
            NDArray array = manager.create(new float[]{feature});
            return new NDList(array);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.singletonOrThrow();
            List<String> classes = List.of("positiv", "negativ"); // Klassen
            return new Classifications(classes, probabilities);
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
} */
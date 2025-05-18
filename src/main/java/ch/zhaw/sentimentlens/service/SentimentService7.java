/*  package ch.zhaw.sentimentlens.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Batchifier;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Service
public class SentimentService7 {

    private Model model;
    private Predictor<String, Classifications> predictor;

    @PostConstruct
    public void init() throws IOException, MalformedModelException {
        try {
            model = Model.newInstance("sentiment-model");

            // Block-Struktur wie beim Training
            Block block = new SequentialBlock()
                .add(Linear.builder().setUnits(128).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(64).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(2).build());

            model.setBlock(block);

            // Modellverzeichnis
            Path modelDir = Paths.get(getClass().getClassLoader().getResource("model").toURI());
            model.load(modelDir, "sentiment-model");

            predictor = model.newPredictor(new SentimentTranslator());
        } catch (URISyntaxException e) {
            throw new IOException("Modellpfad konnte nicht geladen werden", e);
        }
    }

    public String analyzeSentiment(String text) throws TranslateException {
        Classifications classifications = predictor.predict(text);
        Classification best = classifications.best();
        return String.format("Erkanntes Sentiment: %s (%.2f%%)", best.getClassName(), best.getProbability() * 100);
    }

    private static class SentimentTranslator implements Translator<String, Classifications> {
        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            NDManager manager = ctx.getNDManager();
            float feature = input.length() / 5.0f;
            NDArray array = manager.create(new float[]{feature});
            return new NDList(array);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.singletonOrThrow();
            List<String> classes = List.of("positiv", "negativ");
            return new Classifications(classes, probabilities);
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
} */

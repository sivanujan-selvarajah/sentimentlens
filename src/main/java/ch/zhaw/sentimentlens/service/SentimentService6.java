package ch.zhaw.sentimentlens.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ch.zhaw.sentimentlens.translate.SentimentTranslator2;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

@Service
public class SentimentService6 {

    private Model model;
    private Predictor<String, Classifications> predictor;
    private List<String> vocabulary;

    @PostConstruct
    public void init() throws IOException, MalformedModelException {
        try {
            // Modellstruktur wie im Training
            model = Model.newInstance("sentiment-model");
            Block block = new SequentialBlock()
                    .add(Linear.builder().setUnits(256).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(128).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(64).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(2).build());
            model.setBlock(block);

            // Modell laden
            Path modelDir = Paths.get(getClass().getClassLoader().getResource("model").toURI());
            model.load(modelDir, "sentiment-model");

          try (var stream = getClass().getClassLoader().getResourceAsStream("model/vocabulary.txt")) {
            if (stream == null) {
            throw new IOException("vocabulary.txt nicht gefunden!");
    }
    vocabulary = new ArrayList<>();
    try (var scanner = new Scanner(stream)) {
        while (scanner.hasNextLine()) {
            vocabulary.add(scanner.nextLine());
        }
    }
}

            // Predictor mit passendem Translator initialisieren
            predictor = model.newPredictor(new SentimentTranslator2(vocabulary));
        } catch (URISyntaxException e) {
            throw new IOException("Fehler beim Laden des Pfads", e);
        }
    }

    public String analyzeSentiment(String text) throws ai.djl.translate.TranslateException {
        Classifications classifications = predictor.predict(text);
        Classification best = classifications.best();
        return String.format("Erkanntes Sentiment: %s (%.2f%%)", best.getClassName(), best.getProbability() * 100);
    }
}


/* @PostConstruct
public void init() throws IOException, MalformedModelException {
    model = Model.newInstance(); // Keine Engine angeben
    model.load(Paths.get("src/main/resources/model"), "sentiment-model"); // NUR Basisname angeben

    predictor = model.newPredictor(new SentimentTranslator());
} */
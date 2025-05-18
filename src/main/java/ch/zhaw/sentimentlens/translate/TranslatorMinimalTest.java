 /* package ch.zhaw.sentimentlens.translate;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.translate.TranslateException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class TranslatorMinimalTest {

    public static void main(String[] args) throws IOException, TranslateException, MalformedModelException {
        // Vokabular aus Datei laden
        Path vocabPath = Paths.get("src/main/resources/model/vocabulary.txt");
        List<String> vocab = Files.readAllLines(vocabPath);

        // Eingabetext definieren
        String input = "nice good good";

        // Vektor debuggen
        Map<String, Integer> tokenCounts = new HashMap<>();
        for (String token : input.toLowerCase().split("\\W+")) {
            tokenCounts.put(token, tokenCounts.getOrDefault(token, 0) + 1);
        }

        float[] vector = new float[vocab.size()];
        for (int i = 0; i < vocab.size(); i++) {
            vector[i] = tokenCounts.getOrDefault(vocab.get(i), 0);
        }

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create(vector);
            System.out.println("Input-Vektor: " + array);
        }

        // Modell laden mit Block-Struktur
        Model model = Model.newInstance("sentiment-model");

        // ⬇️ WICHTIG: Block-Struktur wie beim Training setzen!
        Block block = new SequentialBlock()
            .add(Linear.builder().setUnits(128).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(64).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(2).build());
        model.setBlock(block);

        model.load(Paths.get("src/main/resources/model"), "sentiment-model");

        // Vorhersage
        try (Predictor<String, Classifications> predictor = model.newPredictor(new SentimentTranslator(vocab))) {
            Classifications result = predictor.predict(input);
            System.out.println("🔍 Modell-Vorhersage: " + result);
        }
    }
} */

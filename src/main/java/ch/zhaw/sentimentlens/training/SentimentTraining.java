package ch.zhaw.sentimentlens.training;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.loss.Loss;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ch.zhaw.sentimentlens.dataset.SentimentDataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class SentimentTraining {

    public static void main(String[] args) throws IOException {
        int batchSize = 32;
        int numEpochs = 15;

        // Dataset laden
        SentimentDataset dataset = new SentimentDataset();
        dataset.load("src/main/resources/sampled_reviews.txt");

        List<String> texts = dataset.getTexts();
        List<Integer> labels = dataset.getLabels();

        try (NDManager manager = NDManager.newBaseManager()) {
            // Vokabular bauen
            Map<String, Integer> wordFreq = new HashMap<>();
            for (String text : texts) {
                for (String word : text.toLowerCase().split("\\W+")) {
                    if (!word.isBlank()) {
                        wordFreq.put(word, wordFreq.getOrDefault(word, 0) + 1);
                    }
                }
            }

            // Top 200 Wörter
            List<String> vocab = wordFreq.entrySet().stream()
                    .sorted((a, b) -> b.getValue() - a.getValue())
                    .limit(200)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            // Feature-Vektoren
            float[][] featureMatrix = new float[texts.size()][vocab.size()];
            for (int i = 0; i < texts.size(); i++) {
                Set<String> words = Arrays.stream(texts.get(i).toLowerCase().split("\\W+"))
                                          .collect(Collectors.toSet());
                for (int j = 0; j < vocab.size(); j++) {
                    featureMatrix[i][j] = words.contains(vocab.get(j)) ? 1.0f : 0.0f;
                }
            }

            var features = manager.create(featureMatrix);
            int[] labelArray = labels.stream().mapToInt(Integer::intValue).toArray();
            var labelNd = manager.create(labelArray);

            var trainDataset = new ArrayDataset.Builder()
                    .setData(features)
                    .optLabels(labelNd)
                    .setSampling(batchSize, true)
                    .build();

            // Modellarchitektur
            Block block = new SequentialBlock()
                    .add(Linear.builder().setUnits(128).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(64).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(2).build());

            Model model = Model.newInstance("sentiment-model");
            model.setBlock(block);

            var lossFunction = Loss.softmaxCrossEntropyLoss();
            var config = new DefaultTrainingConfig(lossFunction)
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.fixed(0.001f))
                            .build())
                    .addTrainingListeners(TrainingListener.Defaults.basic());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new ai.djl.ndarray.types.Shape(batchSize, vocab.size()));

                for (int epoch = 0; epoch < numEpochs; epoch++) {
                    System.out.println("Epoche " + (epoch + 1));
                   
                }
            }

            // Modell & Vokabular speichern
            Path modelDir = Paths.get("src/main/resources/model");
            model.save(modelDir, "sentiment-model");

            String jsonContent = """
            {
              "input": {
                "data": {
                  "shape": [1, 200],
                  "dtype": "float32"
                }
              },
              "output": {
                "output": {
                  "shape": [1, 2],
                  "dtype": "float32"
                }
              },
              "format": "pt"
            }
            """;
            Files.writeString(modelDir.resolve("sentiment-model.json"), jsonContent);
            Files.write(modelDir.resolve("vocabulary.txt"), vocab); // << Wichtig!

            System.out.println("Training abgeschlossen und gespeichert unter: src/main/resources/model");
        }
    }
}
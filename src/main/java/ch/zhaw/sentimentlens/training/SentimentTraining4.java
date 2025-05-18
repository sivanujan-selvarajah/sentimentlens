package ch.zhaw.sentimentlens.training;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.loss.Loss;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
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

public class SentimentTraining4 {

    public static void main(String[] args) throws IOException, TranslateException {
        int batchSize = 32;
        int numEpochs = 30;

        SentimentDataset dataset = new SentimentDataset();
        dataset.load("src/main/resources/sampled_reviews.txt");

        List<String> texts = dataset.getTexts();
        List<Integer> labels = dataset.getLabels();

        try (NDManager manager = NDManager.newBaseManager()) {
            // 1. Vokabular zählen
            Map<String, Integer> docFreq = new HashMap<>();
            List<Set<String>> tokenizedDocs = new ArrayList<>();

            for (String text : texts) {
                Set<String> tokens = Arrays.stream(text.toLowerCase().split("\\W+"))
                                           .filter(token -> !token.isBlank())
                                           .collect(Collectors.toSet());
                tokenizedDocs.add(tokens);
                for (String token : tokens) {
                    docFreq.put(token, docFreq.getOrDefault(token, 0) + 1);
                }
            }

            // 2. Top-N Vokabular (500)
            List<String> vocab = docFreq.entrySet().stream()
                    .sorted((a, b) -> b.getValue() - a.getValue())
                    .limit(1000)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            Map<String, Integer> vocabIndex = new HashMap<>();
            for (int i = 0; i < vocab.size(); i++) {
                vocabIndex.put(vocab.get(i), i);
            }

            // 3. TF-IDF Feature-Matrix
            float[][] tfidfMatrix = new float[texts.size()][vocab.size()];
            for (int i = 0; i < texts.size(); i++) {
                String[] words = texts.get(i).toLowerCase().split("\\W+");
                Map<String, Integer> termFreq = new HashMap<>();
                for (String word : words) {
                    if (!word.isBlank()) {
                        termFreq.put(word, termFreq.getOrDefault(word, 0) + 1);
                    }
                }
                for (String term : termFreq.keySet()) {
                    if (vocabIndex.containsKey(term)) {
                        int index = vocabIndex.get(term);
                        int tf = termFreq.get(term);
                        int df = docFreq.get(term);
                        double idf = Math.log((double) texts.size() / (df + 1));
                        tfidfMatrix[i][index] = (float) (tf * idf);
                    }
                }
            }

            var features = manager.create(tfidfMatrix);
            var labelArray = labels.stream().mapToInt(Integer::intValue).toArray();
            var labelNd = manager.create(labelArray);

            var trainDataset = new ArrayDataset.Builder()
                    .setData(features)
                    .optLabels(labelNd)
                    .setSampling(batchSize, true)
                    .build();

            // 4. Modell-Architektur
            Block block = new SequentialBlock()
                    .add(Linear.builder().setUnits(256).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(128).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(64).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(2).build());

            Model model = Model.newInstance("sentiment-model");
            model.setBlock(block);

            var config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.001f)).build())
                    .addTrainingListeners(TrainingListener.Defaults.basic());

                    try (Trainer trainer = model.newTrainer(config)) {
                        trainer.initialize(new ai.djl.ndarray.types.Shape(batchSize, vocab.size()));
                    
                        for (int epoch = 0; epoch < numEpochs; epoch++) {
                            System.out.println("Epoche " + (epoch + 1));
                    
                            for (Batch batch : trainer.iterateDataset(trainDataset)) {
                                try (batch) {
                                    EasyTrain.trainBatch(trainer, batch);
                                    trainer.step();
                                    System.out.println("Epoch " + epoch + " Loss: " +
                                    trainer.getLoss().evaluate(batch.getLabels(), trainer.forward(batch.getData())).getFloat());
                                }

                            }
                            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
                        }


            Path modelDir = Paths.get("src/main/resources/model");
            model.save(modelDir, "sentiment-model");

            // JSON + Vokabular speichern
            String jsonContent = """
            {
              "input": {
                "data": {
                  "shape": [1, 1000],
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
            Files.write(modelDir.resolve("vocabulary.txt"), vocab);
            System.out.println("✅ TF-IDF Training abgeschlossen!");
            long positives = labels.stream().filter(l -> l == 1).count();
            long negatives = labels.stream().filter(l -> l == 0).count();
            System.out.printf("📊 Label-Verteilung – Positiv: %d, Negativ: %d\n", positives, negatives);
            
        }
    }
}
}
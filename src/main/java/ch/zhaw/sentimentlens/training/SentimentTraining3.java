package ch.zhaw.sentimentlens.training;

import ai.djl.translate.TranslateException; 
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
import java.util.List;

public class SentimentTraining3 {

    public static void main(String[] args) throws IOException, TranslateException {
        int batchSize = 32;
        int numEpochs = 15;

        // Lade das Dataset
        SentimentDataset dataset = new SentimentDataset();
        dataset.load("src/main/resources/sampled_reviews.txt");

        List<String> texts = dataset.getTexts();
        List<Integer> labels = dataset.getLabels();

        try (NDManager manager = NDManager.newBaseManager()) {
            // Feature-Array: Textlänge geteilt durch 5
            float[] featureArray = new float[texts.size()];
            for (int i = 0; i < texts.size(); i++) {
                featureArray[i] = texts.get(i).length() / 5.0f;
            }

            int[] labelArray = labels.stream().mapToInt(Integer::intValue).toArray();

            var features = manager.create(featureArray, new ai.djl.ndarray.types.Shape(featureArray.length, 1));
            var labelNd = manager.create(labelArray);

            // Dataset
            var trainDataset = new ArrayDataset.Builder()
                    .setData(features)
                    .optLabels(labelNd)
                    .setSampling(batchSize, true)
                    .build();

            // Modell
            Block block = new SequentialBlock()
                    .add(Linear.builder().setUnits(128).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(64).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(2).build());

            Model model = Model.newInstance("sentiment-model");
            model.setBlock(block);

            // Trainingskonfiguration
            var lossFunction = Loss.softmaxCrossEntropyLoss();
            var config = new DefaultTrainingConfig(lossFunction)
                    .optOptimizer(Optimizer.adam()
                        .optLearningRateTracker(Tracker.fixed(0.001f))
                        .build())
                    .addTrainingListeners(TrainingListener.Defaults.basic());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new ai.djl.ndarray.types.Shape(batchSize, 1));

                for (int epoch = 0; epoch < numEpochs; epoch++) {
                    System.out.println("Start Epoche: " + (epoch + 1));

                    for (Batch batch : trainer.iterateDataset(trainDataset)) {
                        try (batch) {
                            EasyTrain.trainBatch(trainer, batch);   // ✅ automatisch forward, loss, backward
                            trainer.step();     // ✅ Schritt machen
                        }
                    }
                    System.out.println("Epoche " + (epoch + 1) + " abgeschlossen.");
                }
            }

            // Modell speichern
            Path modelDir = Paths.get("src/main/resources/model");
            model.save(modelDir, "sentiment-model");

            // JSON automatisch schreiben
            String jsonContent = """
            {
              "input": {
                "data": {
                  "shape": [1, 1],
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

            System.out.println("Modell und JSON erfolgreich gespeichert unter: src/main/resources/model");
        }
    }
}
package ch.zhaw.sentimentlens.training;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
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
import ai.djl.nn.norm.Dropout;

import ch.zhaw.sentimentlens.dataset.SentimentDataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class SentimentTraining5 {

    private static final Set<String> STOPWORDS = Set.of(
            "the", "and", "but", "not", "no", "a", "an", "is", "are", "was", "were",
            "this", "that", "it", "of", "in", "on", "for", "to", "with", "at", "by",
            "from", "as", "about", "into", "like", "through", "after", "over", "between",
            "out", "against", "during", "without", "before", "under", "around", "among"
    );

    public static void main(String[] args) throws IOException, TranslateException {
        int batchSize = 32;
        int numEpochs = 30;

        SentimentDataset dataset = new SentimentDataset();
        dataset.load("src/main/resources/sampled_reviews.txt");

        List<String> texts = dataset.getTexts();
        List<Integer> labels = dataset.getLabels();

        try (NDManager manager = NDManager.newBaseManager()) {

            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < texts.size(); i++) indices.add(i);
            Collections.shuffle(indices, new Random(42));

            int trainSize = (int) (0.8 * texts.size());
            List<String> trainTexts = new ArrayList<>();
            List<Integer> trainLabels = new ArrayList<>();
            List<String> testTexts = new ArrayList<>();
            List<Integer> testLabels = new ArrayList<>();

            for (int i = 0; i < texts.size(); i++) {
                if (i < trainSize) {
                    trainTexts.add(texts.get(indices.get(i)));
                    trainLabels.add(labels.get(indices.get(i)));
                } else {
                    testTexts.add(texts.get(indices.get(i)));
                    testLabels.add(labels.get(indices.get(i)));
                }
            }

            Map<String, Integer> docFreq = new HashMap<>();
            List<Set<String>> tokenizedDocs = new ArrayList<>();

            for (String text : trainTexts) {
                Set<String> tokens = Arrays.stream(text.toLowerCase().split("\\W+"))
                        .filter(token -> !token.isBlank() && !STOPWORDS.contains(token))
                        .collect(Collectors.toSet());
                tokenizedDocs.add(tokens);
                for (String token : tokens) {
                    docFreq.put(token, docFreq.getOrDefault(token, 0) + 1);
                }
            }

            docFreq.entrySet().removeIf(e -> e.getValue() < 5);

            List<String> vocab = docFreq.entrySet().stream()
                    .sorted((a, b) -> b.getValue() - a.getValue())
                    .limit(1000)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            Map<String, Integer> vocabIndex = new HashMap<>();
            for (int i = 0; i < vocab.size(); i++) vocabIndex.put(vocab.get(i), i);

            float[][] trainMatrix = buildTfidfMatrix(trainTexts, vocabIndex, docFreq);
            float[][] testMatrix = buildTfidfMatrix(testTexts, vocabIndex, docFreq);

            var trainFeatures = manager.create(trainMatrix);
            var trainLabelNd = manager.create(trainLabels.stream().mapToInt(i -> i).toArray());
            var testFeatures = manager.create(testMatrix);
            var testLabelNd = manager.create(testLabels.stream().mapToInt(i -> i).toArray());

            var trainDataset = new ArrayDataset.Builder().setData(trainFeatures).optLabels(trainLabelNd).setSampling(batchSize, true).build();
            var testDataset = new ArrayDataset.Builder().setData(testFeatures).optLabels(testLabelNd).setSampling(batchSize, false).build();

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
                        }
                    }

                    int correct = 0, total = 0;
                    for (Batch batch : trainer.iterateDataset(testDataset)) {
                        try (batch) {
                            NDArray output = trainer.evaluate(batch.getData()).singletonOrThrow();
                            NDArray predictions = output.argMax(1);
                            NDArray testLabelsNd = batch.getLabels().singletonOrThrow();
                            correct += (int) predictions.eq(testLabelsNd).sum().getLong();
                            total += testLabelsNd.size();
                        }
                    }
                    double accuracy = (double) correct / total;
                    System.out.printf("✅ Accuracy nach Epoche %d: %.2f%%%n", epoch + 1, accuracy * 100);
                }

                model.save(Paths.get("src/main/resources/model"), "sentiment-model");
                Files.write(Paths.get("src/main/resources/model/vocabulary.txt"), vocab);
            }
        }
    }

    private static float[][] buildTfidfMatrix(List<String> texts, Map<String, Integer> vocabIndex, Map<String, Integer> docFreq) {
        float[][] matrix = new float[texts.size()][vocabIndex.size()];
        for (int i = 0; i < texts.size(); i++) {
            String[] words = texts.get(i).toLowerCase().split("\\W+");
            Map<String, Integer> termFreq = new HashMap<>();
            for (String word : words) {
                if (!word.isBlank() && !STOPWORDS.contains(word)) termFreq.put(word, termFreq.getOrDefault(word, 0) + 1);
            }
            for (String term : termFreq.keySet()) {
                if (vocabIndex.containsKey(term)) {
                    int index = vocabIndex.get(term);
                    int tf = termFreq.get(term);
                    int df = docFreq.getOrDefault(term, 1);
                    double idf = Math.log((double) texts.size() / (df + 1));
                    matrix[i][index] = (float) (tf * idf);
                }
            }
        }
        return matrix;
    }
}

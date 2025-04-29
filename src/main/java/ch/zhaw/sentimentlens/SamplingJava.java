package ch.zhaw.sentimentlens;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SamplingJava {

    public static void main(String[] args) {
        String inputFilePath = "src/main/resources/train.ft.txt.bz2"; // Hier liegt deine train.ft.txt.bz2
        String outputFilePath = "src/main/resources/sampled_reviews.txt"; // Neue Datei wird erstellt

        int targetPositive = 1000;
        int targetNegative = 1000;

        List<String> positiveReviews = new ArrayList<>();
        List<String> negativeReviews = new ArrayList<>();

        try (InputStream fileStream = new FileInputStream(inputFilePath);
             BufferedInputStream bufferedStream = new BufferedInputStream(fileStream);
             BZip2CompressorInputStream bzIn = new BZip2CompressorInputStream(bufferedStream);
             BufferedReader reader = new BufferedReader(new InputStreamReader(bzIn, StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("__label__1") && negativeReviews.size() < targetNegative) {
                    negativeReviews.add(line);
                } else if (line.startsWith("__label__2") && positiveReviews.size() < targetPositive) {
                    positiveReviews.add(line);
                }

                if (positiveReviews.size() == targetPositive && negativeReviews.size() == targetNegative) {
                    break; // Fertig!
                }
            }

            // Alles mischen, damit es gemixt ist
            List<String> allSamples = new ArrayList<>();
            allSamples.addAll(positiveReviews);
            allSamples.addAll(negativeReviews);
            Collections.shuffle(allSamples);

            // Ergebnisse speichern
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
                for (String review : allSamples) {
                    writer.write(review);
                    writer.newLine();
                }
            }

            System.out.println("Sampling abgeschlossen!");
            System.out.println("Positive Reviews: " + positiveReviews.size());
            System.out.println("Negative Reviews: " + negativeReviews.size());
            System.out.println("Gespeichert in: " + outputFilePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
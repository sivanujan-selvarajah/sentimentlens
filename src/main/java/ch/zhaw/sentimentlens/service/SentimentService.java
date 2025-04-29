/* package ch.zhaw.sentimentlens.service;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;

import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class SentimentService {

    private final ZooModel<String, Classifications> model;
    private final Predictor<String, Classifications> predictor;

    public SentimentService() throws IOException, ModelException {
        Criteria<String, Classifications> criteria = Criteria.builder()
        .setTypes(String.class, Classifications.class)
        .optModelUrls("djl://ai.djl.huggingface.pytorch/distilbert-base-uncased-finetuned-sst-2-english")
        .optEngine("PyTorch") // wichtig!
        .build();;


       //Criteria<String, Classifications> criteria = Criteria.builder()
                //.optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                //.setTypes(String.class, Classifications.class)
                //.build();

        this.model = ModelZoo.loadModel(criteria);
        this.predictor = model.newPredictor();
    }

    public Classifications predict(String text) throws TranslateException {
        return predictor.predict(text);
    }
} */
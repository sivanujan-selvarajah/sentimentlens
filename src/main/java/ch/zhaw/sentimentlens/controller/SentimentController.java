 /* package ch.zhaw.sentimentlens.controller;

import ch.zhaw.sentimentlens.service.SentimentService;
import ai.djl.modality.Classifications;
import ai.djl.translate.TranslateException;
import ai.djl.ModelException;

import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@CrossOrigin // erlaubt auch Zugriff vom Browser (Frontend später)
public class SentimentController {

    private final SentimentService sentimentService;

    public SentimentController() throws IOException, ModelException {
        this.sentimentService = new SentimentService();
    }


    @PostMapping("/analyze")
public List<Map<String, Object>> analyzeText(@RequestBody String text) throws TranslateException {
    Classifications result = sentimentService.predict(text);

    return result.items().stream().map(c -> {
        Map<String, Object> entry = new HashMap<>();
        entry.put("label", c.getClassName());
        entry.put("probability", c.getProbability());
        return entry;
    }).collect(Collectors.toList());
}

    @GetMapping("/ping")
    public String ping() {
        return "SentimentLens ist online!";
    }
} */
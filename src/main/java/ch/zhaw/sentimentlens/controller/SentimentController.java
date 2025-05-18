package ch.zhaw.sentimentlens.controller;

import ch.zhaw.sentimentlens.service.SentimentService6;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api") // wichtig: Basis-Pfad
@CrossOrigin
public class SentimentController {

    private final SentimentService6 sentimentService6;

    public SentimentController(SentimentService6 sentimentService6) {
        this.sentimentService6 = sentimentService6;
    }

    @PostMapping("/analyze")
    public Map<String, String> analyzeText(@RequestBody String text) {
        try {
            String result = sentimentService6.analyzeSentiment(text);
            return Map.of("result", result);
        } catch (Exception e) {
            return Map.of("error", "Analyse fehlgeschlagen: " + e.getMessage());
        }
    }

    @GetMapping("/ping")
    public String ping() {
        return "SentimentLens ist online!";
    }
}
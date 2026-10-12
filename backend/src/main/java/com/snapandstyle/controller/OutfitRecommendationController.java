package com.snapandstyle.controller;

import com.snapandstyle.service.RecommendationClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/outfits")
public class OutfitRecommendationController {

    private final RecommendationClient recommendationClient;

    public OutfitRecommendationController(
            RecommendationClient recommendationClient) {
        this.recommendationClient = recommendationClient;
    }

    @GetMapping("/recommend")
    public Map<String, Object> recommend(
            @RequestParam(defaultValue = "Men") String gender,
            @RequestParam(defaultValue = "Fall") String season,
            @RequestParam(defaultValue = "college") String occasion,
            @RequestParam(defaultValue = "3") int limit) {

        return recommendationClient.recommend(
                gender, season, occasion, limit);
    }
}
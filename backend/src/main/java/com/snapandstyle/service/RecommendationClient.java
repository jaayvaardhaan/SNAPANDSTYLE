package com.snapandstyle.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class RecommendationClient {

    private final RestClient restClient;

    public RecommendationClient(
            RestClient.Builder builder,
            @Value("${snapandstyle.recommendation-url:http://localhost:8001}")
            String recommendationUrl) {

        this.restClient = builder
                .baseUrl(recommendationUrl)
                .build();
    }

    public Map<String, Object> recommend(
            String gender, String season, String occasion, int limit) {

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("gender", gender);
        request.put("season", season);
        request.put("occasion", occasion);
        request.put("limit", Math.max(1, Math.min(limit, 10)));

        try {
            Map<?, ?> response = restClient.post()
                    .uri("/recommend")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .body(Map.class);

            if (response == null) {
                throw new ResponseStatusException(
                        HttpStatus.BAD_GATEWAY,
                        "Python recommendation API returned no response");
            }

            Map<String, Object> result = new LinkedHashMap<>();
            response.forEach((key, value) ->
                    result.put(String.valueOf(key), value));

            return result;

        } catch (RestClientException e) {
            throw new ResponseStatusException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Could not connect to the Python recommendation API on port 8001.",
                    e);
        }
    }
}
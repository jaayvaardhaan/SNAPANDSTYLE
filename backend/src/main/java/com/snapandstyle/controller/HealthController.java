package com.snapandstyle.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class HealthController {

    @GetMapping("/api/health")
    public Map<String, String> health() {
        Map<String, String> response = new LinkedHashMap<>();

        response.put("status", "UP");
        response.put("service", "SnapAndStyle Backend");
        response.put("message", "Backend is running successfully");

        return response;
    }
}

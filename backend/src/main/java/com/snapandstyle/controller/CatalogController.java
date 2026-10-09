
package com.snapandstyle.controller;

import com.snapandstyle.service.CatalogService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/catalog")
public class CatalogController {

    private final CatalogService catalogService;

    public CatalogController(CatalogService catalogService) {
        this.catalogService = catalogService;
    }

    @GetMapping("/items")
    public Map<String, Object> getItems(
            @RequestParam(required = false) String gender,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String season,
            @RequestParam(required = false) String colour,
            @RequestParam(required = false) String occasion,
            @RequestParam(defaultValue = "20") int limit) {

        List<Map<String, String>> results =
                catalogService.search(
                        gender, category, season,
                        colour, occasion, limit);

        Map<String, Object> response = new LinkedHashMap<>();

        response.put("status", "success");
        response.put("totalCatalogItems",
                catalogService.getItemCount());
        response.put("resultCount", results.size());
        response.put("items", results);

        return response;
    }
}

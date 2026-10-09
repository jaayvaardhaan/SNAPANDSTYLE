package com.snapandstyle.service;

import jakarta.annotation.PostConstruct;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.Reader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Service
public class CatalogService {

    @Value("${snapandstyle.dataset-path:../datasets/snapstyle_items_tagged.csv}")
    private String datasetPath;

    private List<Map<String, String>> items = new ArrayList<>();

    @PostConstruct
    public void loadDataset() {
        Path path = Path.of(datasetPath).toAbsolutePath().normalize();

        try (Reader reader = Files.newBufferedReader(
                    path, StandardCharsets.UTF_8);
             CSVParser parser = CSVParser.parse(
                    reader,
                    CSVFormat.DEFAULT.builder()
                            .setHeader()
                            .setSkipHeaderRecord(true)
                            .build())) {

            List<Map<String, String>> loadedItems = new ArrayList<>();

            for (CSVRecord record : parser) {
                Map<String, String> item = new LinkedHashMap<>();

                for (String column : parser.getHeaderMap().keySet()) {
                    item.put(column, record.get(column));
                }

                loadedItems.add(item);
            }

            items = List.copyOf(loadedItems);

            System.out.println(
                    "SnapAndStyle catalog loaded: "
                    + items.size() + " items from " + path);

        } catch (IOException e) {
            throw new IllegalStateException(
                    "Could not load clothing dataset: " + path, e);
        }
    }

    public List<Map<String, String>> search(
            String gender,
            String category,
            String season,
            String colour,
            String occasion,
            int limit) {

        return items.stream()
                .filter(item -> matches(item.get("gender"), gender))
                .filter(item ->
                        matches(item.get("our_category"), category)
                        || matches(item.get("articleType"), category)
                        || isBlank(category))
                .filter(item -> matches(item.get("season"), season))
                .filter(item -> matches(item.get("baseColour"), colour))
                .filter(item -> matches(item.get("occasion_tags"), occasion))
                .limit(Math.max(1, Math.min(limit, 100)))
                .toList();
    }

    public int getItemCount() {
        return items.size();
    }

    private boolean matches(String actual, String requested) {
        if (isBlank(requested)) {
            return true;
        }

        if (actual == null) {
            return false;
        }

        return actual.toLowerCase(Locale.ROOT)
                .contains(requested.trim().toLowerCase(Locale.ROOT));
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}

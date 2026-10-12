package com.snapandstyle.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CatalogServiceTest {
    @TempDir
    Path folder;

    private CatalogService catalog;

    @BeforeEach
    void loadCatalog() throws Exception {
        Path csv = folder.resolve("catalog.csv");
        Files.writeString(csv, String.join("\n",
                "id,gender,our_category,articleType,season,baseColour,occasion_tags",
                "1,Men,topwear,Shirts,Summer,Blue,\"office,college\"",
                "2,Women,topwear,Tops,Summer,Blue,college",
                "3,Unisex,footwear,Casual Shoes,Winter,Navy Blue,\"casual_outing,college\""));
        catalog = new CatalogService();
        ReflectionTestUtils.setField(catalog, "datasetPath", csv.toString());
        catalog.loadDataset();
    }

    private List<String> ids(String gender, String category, String season,
                             String colour, String occasion) {
        return catalog.search(gender, category, season, colour, occasion, 20)
                .stream().map(item -> item.get("id")).toList();
    }

    @Test
    void genderRequiresExactMatch() {
        assertEquals(List.of("1"), ids("Men", null, null, null, null));
        assertEquals(List.of("2"), ids("Women", null, null, null, null));
        assertEquals(List.of("3"), ids("Unisex", null, null, null, null));
    }

    @Test
    void categoryMatchesExactCategoryOrArticle() {
        assertEquals(List.of("1", "2"), ids(null, "topwear", null, null, null));
        assertEquals(List.of("3"), ids(null, "Casual Shoes", null, null, null));
        assertEquals(List.of(), ids(null, "Casual", null, null, null));
    }

    @Test
    void seasonAndColourRequireExactMatch() {
        assertEquals(List.of("1", "2"), ids(null, null, "Summer", "Blue", null));
        assertEquals(List.of(), ids(null, null, "Sum", null, null));
        assertEquals(List.of(), ids(null, null, null, "Navy", null));
    }

    @Test
    void occasionRequiresWholeTag() {
        assertEquals(List.of("1", "2", "3"), ids(null, null, null, null, "COLLEGE"));
        assertEquals(List.of(), ids(null, null, null, null, "lege"));
        assertEquals(List.of("3"), ids(null, null, null, null, "casual_outing"));
    }
}

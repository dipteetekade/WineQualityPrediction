package com.example;

import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.io.IOException;

public class TrainApp {
    public static void main(String[] args) {
        // Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .getOrCreate();

        // Load training data from local filesystem
        Dataset<Row> trainingData = spark.read()
                .option("header", "true")      // First row contains column names
                .option("inferSchema", "true") // Automatically infer data types
                .option("delimiter", ";")      // Specify semicolon as the delimiter
                .option("quote", "\"")         // Handle quoted column names
                .csv("file:///home/ubuntu/datasets/TrainingDataset.csv");

        // Load validation data from local filesystem
        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ";")
                .option("quote", "\"")
                .csv("file:///home/ubuntu/datasets/ValidationDataset.csv");

        // Print schema for debugging
        System.out.println("Training Data Schema:");
        trainingData.printSchema();
        System.out.println("Validation Data Schema:");
        validationData.printSchema();

        // Rename columns to match expected names by removing extra quotes
        Dataset<Row> trainingDataRenamed = renameColumns(trainingData);
        Dataset<Row> validationDataRenamed = renameColumns(validationData);

        // Assemble features into a single vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"volatile acidity", "citric acid",
                        "residual sugar", "chlorides", "free sulfur dioxide",
                        "total sulfur dioxide", "density", "sulphates", "alcohol"})
                .setOutputCol("features");

        Dataset<Row> trainingTransformed = assembler.transform(trainingDataRenamed);

        // Train Logistic Regression model
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features");
        LogisticRegressionModel model = lr.fit(trainingTransformed);

        // Evaluate the model
        Dataset<Row> validationTransformed = assembler.transform(validationDataRenamed);
        Dataset<Row> predictions = model.transform(validationTransformed);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1Score);

        // Save the model
        try {
            model.save("file:///home/ubuntu/models/wine-quality-model");
        } catch (IOException e) {
            System.err.println("Failed to save the model: " + e.getMessage());
            e.printStackTrace();
        }

        // Stop Spark session
        spark.stop();
    }

    /**
     * Rename columns by removing extra quotes from column names.
     */
    private static Dataset<Row> renameColumns(Dataset<Row> dataset) {
        for (String column : dataset.columns()) {
            String cleanedColumn = column.replace("\"", "").trim();
            dataset = dataset.withColumnRenamed(column, cleanedColumn);
        }
        return dataset;
    }
}

package com.example;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import java.io.PrintWriter;

public class PredictionApp {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: PredictApp <model-path> <input-csv>");
            System.exit(1);
        }

        String modelPath = args[0];
        String inputCsvPath = args[1];

        SparkSession spark = SparkSession.builder()
            .appName("WineQualityPrediction")
            .getOrCreate();

        Dataset<Row> rawTestData = spark.read()
            .option("header", "true")
            .option("inferSchema", "true")
            .option("delimiter", ";")
            .csv(inputCsvPath);

        Dataset<Row> testData = cleanHeadings(rawTestData);

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{
                "volatile_acidity", "citric_acid", "residual_sugar", 
                "chlorides", "free_sulfur_dioxide", 
                "total_sulfur_dioxide", "density", 
                "sulphates", "alcohol"})
            .setOutputCol("features");

        Dataset<Row> transformedTestData = assembler.transform(testData);
        LogisticRegressionModel model = LogisticRegressionModel.load(modelPath);
        Dataset<Row> predictions = model.transform(transformedTestData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("quality")
            .setPredictionCol("prediction")
            .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);

        // Print F1 Score to console
        System.out.println("F1 Score: " + f1Score);

        // Write predictions to a file
        predictions.select("quality", "prediction")
            .write()
            .option("header", "true")
            .csv("predictions_output");

        // Write the F1 score to a separate file
        try (PrintWriter writer = new PrintWriter("f1_score.txt", "UTF-8")) {
            writer.println("F1 Score: " + f1Score);
        } catch (Exception e) {
            e.printStackTrace();
        }

        spark.stop();
    }

    private static Dataset<Row> cleanHeadings(Dataset<Row> rawDataset) {
        StructType schema = rawDataset.schema();
        String[] cleanedColumnNames = schema.fieldNames();

        for (int i = 0; i < cleanedColumnNames.length; i++) {
            cleanedColumnNames[i] = cleanedColumnNames[i]
                .replaceAll("\"", "")
                .trim()
                .replace(" ", "_");
        }

        for (int i = 0; i < cleanedColumnNames.length; i++) {
            rawDataset = rawDataset.withColumnRenamed(schema.fieldNames()[i], cleanedColumnNames[i]);
        }

        return rawDataset;
    }
}
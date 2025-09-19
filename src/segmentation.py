import os
import matplotlib # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

from utils import create_spark_session
from pyspark.sql.functions import (col, min, max, countDistinct, sum as spark_sum,
                                   datediff, lit, log1p, when, dayofweek, hour)
from pyspark.ml import Pipeline
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import (VectorAssembler, StandardScaler, StringIndexer,
                                OneHotEncoder, PCA)

INPUT_DATA_PATH = "data/online_retail.parquet"
OUTPUT_DATA_PATH = "output/segmented_customers_gmm.csv"
SILHOUETTE_PLOT_PATH = "output/silhouette_plot.png"
PCA_PLOT_PATH = "output/pca_clusters.png"

def engineer_features(data):
    print("Step 1: Engineering RFM and behavioral features...")
    rfm_data = data.groupBy("CustomerID").agg(
        min("InvoiceDate").alias("FirstPurchaseDate"),
        max("InvoiceDate").alias("LastPurchaseDate"),
        countDistinct("InvoiceNo").alias("Frequency"),
        spark_sum("TotalAmount").alias("Monetary"),
        countDistinct("StockCode").alias("ProductDiversity"),
        spark_sum("Quantity").alias("TotalQuantity")
    )

    max_date = data.agg(max("InvoiceDate")).collect()[0][0]
    rfm_data = rfm_data.withColumn("Recency", datediff(lit(max_date), col("LastPurchaseDate"))) \
                       .withColumn("Tenure", datediff(col("LastPurchaseDate"), col("FirstPurchaseDate"))) \
                       .withColumn("LastPurchaseDayOfWeek", dayofweek(col("LastPurchaseDate"))) \
                       .withColumn("LastPurchaseHour", hour(col("LastPurchaseDate")))
    
    rfm_data = rfm_data.withColumn("Tenure", when(col("Tenure") == 0, 1).otherwise(col("Tenure")))

    skewed_cols = ["Recency", "Frequency", "Monetary", "Tenure", "ProductDiversity", "TotalQuantity"]
    for col_name in skewed_cols:
        rfm_data = rfm_data.withColumn(f"log_{col_name}", log1p(col(col_name)))

    rfm_data = rfm_data.withColumn("AvgOrderValue", col("Monetary") / col("Frequency")) \
                       .withColumn("AvgDaysBetweenPurchases", col("Tenure") / col("Frequency")) \
                       .withColumn("IsReturning", when(col("Frequency") > 1, 1).otherwise(0))

    country_df = data.select("CustomerID", "Country").distinct()
    rfm_data = rfm_data.join(country_df, on="CustomerID", how="left").na.fill({"Country": "Unknown"})
    country_counts = rfm_data.groupBy("Country").count().orderBy(col("count").desc())
    top_10_countries = [row['Country'] for row in country_counts.limit(10).collect()]
    rfm_data = rfm_data.withColumn("CountryGroup", when(col("Country").isin(top_10_countries), col("Country")).otherwise(lit("Other")))
    
    return rfm_data


def assemble_and_scale_features(df):
    print("Step 2: Assembling and scaling feature vector...")
    log_feature_cols = [f"log_{c}" for c in ["Recency", "Frequency", "Monetary", "Tenure", "ProductDiversity", "TotalQuantity"]]
    engineered_cols = ["AvgOrderValue", "AvgDaysBetweenPurchases", "IsReturning"]
    time_based_cols = ["LastPurchaseDayOfWeek", "LastPurchaseHour"]
    categorical_cols_to_encode = ["CountryGroup"]
    
    feature_cols = log_feature_cols + engineered_cols + time_based_cols
    
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}Index", handleInvalid="keep") for c in categorical_cols_to_encode]
    encoders = [OneHotEncoder(inputCols=[f"{c}Index"], outputCols=[f"{c}Vec"]) for c in categorical_cols_to_encode]
    assembler = VectorAssembler(inputCols=feature_cols + [f"{c}Vec" for c in categorical_cols_to_encode], outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    model = pipeline.fit(df)
    return model.transform(df)


def find_optimal_clusters(df, plot_path):
    print(f"Step 3: Finding optimal K and saving plot to {plot_path}...")
    silhouette_scores = []
    K_range = range(3, 9)
    
    for k in K_range:
        gmm = GaussianMixture(featuresCol="scaled_features", k=k, seed=42)
        model = gmm.fit(df)
        predictions = model.transform(df)
        evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette")
        silhouette = evaluator.evaluate(predictions)
        silhouette_scores.append(silhouette)
        print(f"K={k}, Silhouette Score={silhouette:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouette_scores, marker='o', color="green")
    plt.title("Silhouette Scores for GMM")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    return K_range[np.argmax(silhouette_scores)]


def train_final_model(df, k):
    print(f"Step 4: Training final GMM model with best K={k}...")
    gmm = GaussianMixture(featuresCol="scaled_features", k=k, seed=42, maxIter=100)
    model = gmm.fit(df)
    return model.transform(df)


def profile_and_print_clusters(df):
    print("Step 5: Profiling customer segments...")
    profile_cols = ["Recency", "Frequency", "Monetary", "Tenure", "AvgOrderValue", "prediction"]
    profile_summary = df.groupBy("prediction").mean(*[c for c in profile_cols if c != 'prediction']).orderBy("prediction")
    
    for col_name in profile_summary.columns[1:]:
        profile_summary = profile_summary.withColumnRenamed(f"avg({col_name})", col_name)
    
    profile_pd = profile_summary.toPandas().set_index('prediction')
    print("\n--- Cluster Profile Summary (Averages) ---")
    print(profile_pd.to_string(float_format="%.2f"))
    print("------------------------------------------\n")


def visualize_and_save_pca(df, plot_path):
    print(f"Step 6: Visualizing clusters with PCA and saving plot to {plot_path}...")
    pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
    rfm_pca = pca.fit(df).transform(df)
    
    pandas_df = rfm_pca.select("pca_features", "prediction").toPandas()
    pandas_df['x'] = pandas_df['pca_features'].apply(lambda v: float(v[0]))
    pandas_df['y'] = pandas_df['pca_features'].apply(lambda v: float(v[1]))
    
    plt.figure(figsize=(10, 8))
    for cluster in sorted(pandas_df['prediction'].unique()):
        subset = pandas_df[pandas_df['prediction'] == cluster]
        plt.scatter(subset['x'], subset['y'], label=f"Cluster {cluster}", alpha=0.7)
    
    plt.title("Customer Segments (PCA 2D Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


def main():
    spark = create_spark_session()
    
    # Load and process data.
    raw_data = spark.read.parquet(INPUT_DATA_PATH)
    featured_data = engineer_features(raw_data)
    scaled_data = assemble_and_scale_features(featured_data)
    
    # Find best k and train model.
    best_k = find_optimal_clusters(scaled_data, SILHOUETTE_PLOT_PATH)
    clustered_data = train_final_model(scaled_data, best_k)
    
    # Analyze and save results.
    profile_and_print_clusters(clustered_data)
    visualize_and_save_pca(clustered_data, PCA_PLOT_PATH)
    
    print(f"Step 7: Saving final segmented data to {OUTPUT_DATA_PATH}...")
    output_cols = ["CustomerID", "Recency", "Frequency", "Monetary", "CountryGroup", "prediction"]
    clustered_data.select(output_cols) \
                  .coalesce(1) \
                  .write \
                  .mode("overwrite") \
                  .option("header", "true") \
                  .csv(OUTPUT_DATA_PATH)
    
    print("\nPipeline finished successfully!")
    spark.stop()


if __name__ == "__main__":
    main()
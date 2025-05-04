from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, count, countDistinct, mean, stddev, length, regexp_replace
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.stat import Correlation
from pyspark.sql.types import StringType, NumericType, DateType, TimestampType
from typing import List, Dict, Any
import numpy as np
import yaml
import json

class AnomalyDetector:
    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        self.spark = spark
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def detect_statistical_anomalies(self, df: DataFrame, column: str) -> DataFrame:
        """Detect anomalies using statistical methods."""
        col_type = df.schema[column].dataType
        
        if isinstance(col_type, NumericType):
            # Z-score based anomaly detection
            stats = df.select(
                mean(col(column)).alias('mean'),
                stddev(col(column)).alias('stddev')
            ).collect()[0]
            
            mean_val = stats['mean']
            stddev_val = stats['stddev']
            
            df = df.withColumn(
                'z_score',
                (col(column) - mean_val) / stddev_val
            )
            
            # IQR based anomaly detection
            q1 = df.approxQuantile(column, [0.25], 0.01)[0]
            q3 = df.approxQuantile(column, [0.75], 0.01)[0]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            df = df.withColumn(
                'is_anomaly',
                when(
                    (col(column) < lower_bound) | (col(column) > upper_bound) |
                    (abs(col('z_score')) > self.config['anomaly']['threshold']),
                    lit(True)
                ).otherwise(lit(False))
            )
            
        elif isinstance(col_type, StringType):
            # String length anomaly detection
            length_stats = df.select(
                mean(length(col(column))).alias('mean_length'),
                stddev(length(col(column))).alias('stddev_length')
            ).collect()[0]
            
            mean_len = length_stats['mean_length']
            stddev_len = length_stats['stddev_length']
            
            df = df.withColumn(
                'length_z_score',
                (length(col(column)) - mean_len) / stddev_len
            )
            
            # Pattern anomaly detection
            df = df.withColumn(
                'is_anomaly',
                when(
                    (abs(col('length_z_score')) > self.config['anomaly']['threshold']) |
                    (col(column).isNull()) |
                    (trim(col(column)) == ''),
                    lit(True)
                ).otherwise(lit(False))
            )
            
        elif isinstance(col_type, (DateType, TimestampType)):
            # Date anomaly detection
            df = df.withColumn(
                'days_from_mean',
                (col(column) - mean(col(column))).cast('long')
            )
            
            df = df.withColumn(
                'is_anomaly',
                when(
                    abs(col('days_from_mean')) > self.config['anomaly']['lookback_days'],
                    lit(True)
                ).otherwise(lit(False))
            )
        
        return df
    
    def detect_ml_anomalies(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """Detect anomalies using machine learning methods."""
        # Prepare features
        feature_columns = []
        
        for column in columns:
            col_type = df.schema[column].dataType
            
            if isinstance(col_type, NumericType):
                feature_columns.append(column)
            elif isinstance(col_type, StringType):
                # Convert categorical columns to numeric
                indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index")
                df = indexer.fit(df).transform(df)
                feature_columns.append(f"{column}_index")
        
        # Create feature vector
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )
        df = assembler.transform(df)
        
        # Use multiple clustering methods
        # K-means
        kmeans = KMeans(
            k=2,
            seed=42,
            featuresCol="features"
        )
        kmeans_model = kmeans.fit(df)
        df = kmeans_model.transform(df).withColumnRenamed("prediction", "kmeans_prediction")
        
        # Bisecting K-means
        bkm = BisectingKMeans(
            k=2,
            seed=42,
            featuresCol="features"
        )
        bkm_model = bkm.fit(df)
        df = bkm_model.transform(df).withColumnRenamed("prediction", "bkm_prediction")
        
        # Calculate anomaly score based on distance to cluster centers
        df = df.withColumn(
            'anomaly_score',
            when(
                (col('kmeans_prediction') == 1) | (col('bkm_prediction') == 1),
                lit(1.0)
            ).otherwise(lit(0.0))
        )
        
        # Mark anomalies
        df = df.withColumn(
            'is_anomaly',
            when(col('anomaly_score') > self.config['anomaly']['threshold'], lit(True))
            .otherwise(lit(False))
        )
        
        return df
    
    def detect_correlation_anomalies(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """Detect anomalies based on feature correlations."""
        # Calculate correlation matrix
        numeric_cols = [col for col in columns if isinstance(df.schema[col].dataType, NumericType)]
        corr_matrix = Correlation.corr(df, numeric_cols, 'pearson').collect()[0][0]
        
        # Convert correlation matrix to DataFrame
        corr_df = self.spark.createDataFrame(
            [(i, j, float(corr_matrix[i, j])) 
             for i in range(len(numeric_cols)) 
             for j in range(len(numeric_cols))],
            ['col1_idx', 'col2_idx', 'correlation']
        )
        
        # Find highly correlated features
        high_corr = corr_df.filter(
            (abs(col('correlation')) > 0.8) & 
            (col('col1_idx') != col('col2_idx'))
        )
        
        # Calculate expected values based on correlations
        for row in high_corr.collect():
            col1 = numeric_cols[row['col1_idx']]
            col2 = numeric_cols[row['col2_idx']]
            
            # Calculate expected value based on correlation
            df = df.withColumn(
                f'expected_{col2}',
                mean(col(col2)) + (col(col1) - mean(col(col1))) * row['correlation']
            )
            
            # Mark anomalies based on deviation from expected value
            df = df.withColumn(
                f'is_anomaly_{col2}',
                when(
                    abs(col(col2) - col(f'expected_{col2}')) > 
                    (stddev(col(col2)) * self.config['anomaly']['threshold']),
                    lit(True)
                ).otherwise(lit(False))
            )
        
        return df
    
    def save_anomaly_results(self, results: Dict[str, Any]):
        """Save anomaly detection results to Delta table as a single JSON record."""
        catalog = self.config['delta']['catalog']
        schema = self.config['delta']['schema']
        table = self.config['delta']['anomaly_table']
        
        # Convert results to DataFrame with a single row
        results_df = self.spark.createDataFrame([(
            results['table_name'],
            results['database'],
            results['detection_date'],
            json.dumps(results['anomalies'])
        )], ['table_name', 'database', 'detection_date', 'anomaly_data'])
        
        # Write to Delta table
        results_df.write.format("delta").mode("append").saveAsTable(
            f"{catalog}.{schema}.{table}"
        ) 
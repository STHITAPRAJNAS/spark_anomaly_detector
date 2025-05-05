from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, window, count, mean, stddev, expr
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, MinMaxScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.stat import Correlation
from pyspark.sql.types import StringType, NumericType, DateType, TimestampType
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from prophet import Prophet
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import yaml
import json
from scipy import stats

class AdvancedAnomalyDetector:
    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        self.spark = spark
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def extract_time_series_features(self, df: DataFrame, column: str, time_column: str) -> Dict[str, Any]:
        """Extract time series features using tsfresh."""
        # Convert to pandas for tsfresh
        pdf = df.select(time_column, column).toPandas()
        pdf[time_column] = pd.to_datetime(pdf[time_column])
        
        # Extract features
        extracted_features = extract_features(
            pdf,
            column_id=time_column,
            column_value=column,
            default_fc_parameters=EfficientFCParameters()
        )
        
        return extracted_features.to_dict()
    
    def detect_temporal_anomalies(self, df: DataFrame, column: str, time_column: str) -> DataFrame:
        """Detect anomalies using time series analysis."""
        # Convert to pandas for Prophet
        pdf = df.select(time_column, column).toPandas()
        pdf.columns = ['ds', 'y']
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(pdf)
        
        # Make predictions
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        # Calculate residuals
        pdf['residual'] = pdf['y'] - forecast['yhat']
        pdf['residual_std'] = (pdf['residual'] - pdf['residual'].mean()) / pdf['residual'].std()
        
        # Mark anomalies
        threshold = self.config['anomaly']['threshold']
        pdf['is_anomaly'] = abs(pdf['residual_std']) > threshold
        
        # Convert back to Spark DataFrame
        return self.spark.createDataFrame(pdf)
    
    def detect_multivariate_anomalies(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """Detect anomalies using multivariate methods."""
        # Prepare features
        assembler = VectorAssembler(
            inputCols=columns,
            outputCol="features"
        )
        df = assembler.transform(df)
        
        # Scale features
        scaler = MinMaxScaler(
            inputCol="features",
            outputCol="scaled_features"
        )
        df = scaler.fit(df).transform(df)
        
        # Convert to pandas for PyOD
        pdf = df.select("scaled_features").toPandas()
        X = np.array([row[0].toArray() for row in pdf.values])
        
        # Initialize multiple anomaly detection models
        models = {
            'isolation_forest': IForest(contamination=0.1),
            'local_outlier_factor': LOF(contamination=0.1),
            'copod': COPOD(contamination=0.1)
        }
        
        # Fit models and get anomaly scores
        anomaly_scores = {}
        for name, model in models.items():
            model.fit(X)
            anomaly_scores[name] = model.decision_scores_
        
        # Combine scores
        combined_scores = np.mean(list(anomaly_scores.values()), axis=0)
        
        # Convert back to Spark DataFrame
        pdf['anomaly_score'] = combined_scores
        pdf['is_anomaly'] = pdf['anomaly_score'] > np.percentile(combined_scores, 90)
        
        return self.spark.createDataFrame(pdf)
    
    def detect_distribution_anomalies(self, df: DataFrame, column: str) -> DataFrame:
        """Detect anomalies using distribution analysis."""
        # Convert to pandas for distribution analysis
        pdf = df.select(column).toPandas()
        
        # Fit multiple distributions
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'exponential': stats.expon,
            'gamma': stats.gamma
        }
        
        best_fit = None
        best_score = float('inf')
        
        for name, dist in distributions.items():
            try:
                params = dist.fit(pdf[column])
                score = stats.kstest(pdf[column], name, args=params).statistic
                if score < best_score:
                    best_score = score
                    best_fit = (name, params)
            except:
                continue
        
        # Calculate anomaly scores based on best fit
        if best_fit:
            name, params = best_fit
            dist = distributions[name]
            pdf['anomaly_score'] = 1 - dist.cdf(pdf[column], *params)
            pdf['is_anomaly'] = pdf['anomaly_score'] > self.config['anomaly']['threshold']
        
        return self.spark.createDataFrame(pdf)
    
    def detect_pattern_anomalies(self, df: DataFrame, column: str) -> DataFrame:
        """Detect anomalies in patterns and sequences."""
        if isinstance(df.schema[column].dataType, StringType):
            # Pattern analysis for string columns
            df = df.withColumn(
                'pattern_score',
                when(
                    col(column).rlike('^[0-9]+$'), lit(1.0)
                ).when(
                    col(column).rlike('^[a-zA-Z]+$'), lit(0.5)
                ).when(
                    col(column).rlike('^[a-zA-Z0-9]+$'), lit(0.3)
                ).otherwise(lit(0.0))
            )
            
            # Length analysis
            df = df.withColumn(
                'length_score',
                (length(col(column)) - mean(length(col(column)))) / stddev(length(col(column)))
            )
            
            # Combine scores
            df = df.withColumn(
                'anomaly_score',
                (col('pattern_score') + abs(col('length_score'))) / 2
            )
            
            # Mark anomalies
            df = df.withColumn(
                'is_anomaly',
                when(col('anomaly_score') > self.config['anomaly']['threshold'], lit(True))
                .otherwise(lit(False))
            )
        
        return df
    
    def detect_correlation_anomalies(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """Enhanced correlation-based anomaly detection."""
        # Calculate correlation matrix
        numeric_cols = [col for col in columns if isinstance(df.schema[col].dataType, NumericType)]
        corr_matrix = Correlation.corr(df, numeric_cols, 'pearson').collect()[0][0]
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_matrix[i, j]))
        
        # Calculate expected values and anomalies for each pair
        for col1, col2, corr in high_corr_pairs:
            # Calculate expected value based on correlation
            df = df.withColumn(
                f'expected_{col2}',
                mean(col(col2)) + (col(col1) - mean(col(col1))) * corr
            )
            
            # Calculate deviation from expected
            df = df.withColumn(
                f'deviation_{col2}',
                abs(col(col2) - col(f'expected_{col2}')) / stddev(col(col2))
            )
            
            # Mark anomalies
            df = df.withColumn(
                f'is_anomaly_{col2}',
                when(col(f'deviation_{col2}') > self.config['anomaly']['threshold'], lit(True))
                .otherwise(lit(False))
            )
        
        return df
    
    def save_anomaly_results(self, results: Dict[str, Any]):
        """Save advanced anomaly detection results to Delta table."""
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
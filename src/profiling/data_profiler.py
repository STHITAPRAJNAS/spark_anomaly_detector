from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, countDistinct, min, max, mean, stddev, approx_percentile, length, regexp_replace, trim, when, lit, to_json, struct
from pyspark.sql.types import StringType, NumericType, DateType, TimestampType, StructType, StructField, StringType as SparkStringType
from typing import List, Dict, Any
import yaml
import os
import json

class DataProfiler:
    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        self.spark = spark
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def calculate_metrics(self, df: DataFrame, column: str) -> Dict[str, Any]:
        """Calculate profiling metrics for a specific column."""
        metrics = {}
        col_type = df.schema[column].dataType
        
        # Basic metrics for all types
        metrics['count'] = df.select(count(col(column))).collect()[0][0]
        metrics['distinct_count'] = df.select(countDistinct(col(column))).collect()[0][0]
        metrics['null_count'] = df.filter(col(column).isNull()).count()
        metrics['null_percentage'] = (metrics['null_count'] / metrics['count']) * 100 if metrics['count'] > 0 else 0
        
        # Type-specific metrics
        if isinstance(col_type, NumericType):
            # Numeric metrics
            metrics['min'] = df.select(min(col(column))).collect()[0][0]
            metrics['max'] = df.select(max(col(column))).collect()[0][0]
            metrics['mean'] = df.select(mean(col(column))).collect()[0][0]
            metrics['stddev'] = df.select(stddev(col(column))).collect()[0][0]
            metrics['variance'] = metrics['stddev'] ** 2 if metrics['stddev'] is not None else None
            
            # Calculate percentiles
            percentiles = self.config['profiling']['metrics']['percentiles']
            metrics['percentiles'] = df.select(
                approx_percentile(col(column), percentiles, 0.01)
            ).collect()[0][0]
            
            # Skewness and Kurtosis
            metrics['skewness'] = df.select(
                (mean(col(column)) - min(col(column))) / stddev(col(column))
            ).collect()[0][0]
            
        elif isinstance(col_type, StringType):
            # String metrics
            metrics['min_length'] = df.select(min(length(col(column)))).collect()[0][0]
            metrics['max_length'] = df.select(max(length(col(column)))).collect()[0][0]
            metrics['avg_length'] = df.select(mean(length(col(column)))).collect()[0][0]
            
            # Pattern analysis
            metrics['numeric_count'] = df.filter(col(column).rlike('^[0-9]+$')).count()
            metrics['alpha_count'] = df.filter(col(column).rlike('^[a-zA-Z]+$')).count()
            metrics['alphanumeric_count'] = df.filter(col(column).rlike('^[a-zA-Z0-9]+$')).count()
            
            # Top values for categorical data
            top_values = df.groupBy(column).count().orderBy('count', ascending=False).limit(10)
            metrics['top_values'] = {row[column]: row['count'] for row in top_values.collect()}
            
        elif isinstance(col_type, (DateType, TimestampType)):
            # Date/Time metrics
            metrics['min_date'] = df.select(min(col(column))).collect()[0][0]
            metrics['max_date'] = df.select(max(col(column))).collect()[0][0]
            metrics['date_range_days'] = (metrics['max_date'] - metrics['min_date']).days if metrics['max_date'] and metrics['min_date'] else None
            
            # Day of week distribution
            day_dist = df.select(col(column).dayofweek().alias('day')).groupBy('day').count()
            metrics['day_of_week_distribution'] = {row['day']: row['count'] for row in day_dist.collect()}
        
        # Data quality metrics
        metrics['completeness'] = 1 - (metrics['null_count'] / metrics['count']) if metrics['count'] > 0 else 0
        metrics['uniqueness'] = metrics['distinct_count'] / metrics['count'] if metrics['count'] > 0 else 0
        
        return metrics
    
    def profile_table(self, table_name: str, database: str, time_grain: str) -> Dict[str, Any]:
        """Profile a table and return results as a dictionary."""
        df = self.spark.table(f"{database}.{table_name}")
        
        # Get all columns to profile
        columns = df.columns
        
        # Initialize results dictionary
        results = {
            'table_name': table_name,
            'database': database,
            'time_grain': time_grain,
            'profiling_date': self.spark.sql("SELECT current_timestamp()").collect()[0][0],
            'columns': {}
        }
        
        # Calculate metrics for each column
        for column in columns:
            results['columns'][column] = self.calculate_metrics(df, column)
        
        return results
    
    def save_profiling_results(self, results: Dict[str, Any]):
        """Save profiling results to Delta table as a single JSON record."""
        catalog = self.config['delta']['catalog']
        schema = self.config['delta']['schema']
        table = self.config['delta']['profiling_table']
        
        # Convert results to DataFrame with a single row
        results_df = self.spark.createDataFrame([(
            results['table_name'],
            results['database'],
            results['time_grain'],
            results['profiling_date'],
            json.dumps(results['columns'])
        )], ['table_name', 'database', 'time_grain', 'profiling_date', 'profile_data'])
        
        # Write to Delta table
        results_df.write.format("delta").mode("append").saveAsTable(
            f"{catalog}.{schema}.{table}"
        ) 
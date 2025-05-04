from pyspark.sql import SparkSession
from profiling.data_profiler import DataProfiler
from anomaly.anomaly_detector import AnomalyDetector
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import os
import time
from typing import Dict, Any

def create_spark_session():
    """Create and configure Spark session."""
    return SparkSession.builder \
        .appName("DataQualityAnomalyDetection") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

def load_table_config():
    """Load table configuration."""
    with open("config/tables.yaml", 'r') as f:
        return yaml.safe_load(f)

def process_table(spark: SparkSession, table_config: Dict[str, Any], max_retries: int = 3):
    """Process a single table with retry logic."""
    table_name = table_config['name']
    database = table_config['database']
    time_grain = table_config['time_grain']
    anomaly_fields = table_config['anomaly_fields']
    
    profiler = DataProfiler(spark)
    anomaly_detector = AnomalyDetector(spark)
    
    for attempt in range(max_retries):
        try:
            print(f"Processing table: {database}.{table_name} (Attempt {attempt + 1})")
            
            # Profile the table
            profiling_results = profiler.profile_table(table_name, database, time_grain)
            profiler.save_profiling_results(profiling_results)
            
            # Load the table data
            df = spark.table(f"{database}.{table_name}")
            
            # Prepare anomaly results
            anomaly_results = {
                'table_name': table_name,
                'database': database,
                'detection_date': spark.sql("SELECT current_timestamp()").collect()[0][0],
                'anomalies': {}
            }
            
            # Detect anomalies for each field
            for field in anomaly_fields:
                print(f"Detecting anomalies for field: {field}")
                
                # Statistical anomaly detection
                stat_results = anomaly_detector.detect_statistical_anomalies(df, field)
                stat_anomalies = stat_results.filter(col('is_anomaly')).collect()
                anomaly_results['anomalies'][field] = {
                    'statistical': [row.asDict() for row in stat_anomalies]
                }
                
                # Machine learning anomaly detection
                ml_results = anomaly_detector.detect_ml_anomalies(df, [field])
                ml_anomalies = ml_results.filter(col('is_anomaly')).collect()
                anomaly_results['anomalies'][field]['ml'] = [row.asDict() for row in ml_anomalies]
            
            # Detect correlation-based anomalies
            if len(anomaly_fields) > 1:
                print("Detecting correlation-based anomalies")
                corr_results = anomaly_detector.detect_correlation_anomalies(df, anomaly_fields)
                corr_anomalies = corr_results.filter(col('is_anomaly')).collect()
                anomaly_results['anomalies']['correlation'] = [row.asDict() for row in corr_anomalies]
            
            # Save anomaly results
            anomaly_detector.save_anomaly_results(anomaly_results)
            
            return True
            
        except Exception as e:
            print(f"Error processing table {database}.{table_name}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Max retries reached for table {database}.{table_name}")
                return False

def main():
    # Initialize Spark session
    spark = create_spark_session()
    
    # Load configurations
    table_config = load_table_config()
    
    # Configure parallel processing
    max_workers = min(len(table_config['tables']), 4)  # Limit concurrent processing
    batch_size = 2  # Process tables in batches
    
    # Process tables in parallel with controlled concurrency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_table, spark, table): table['name']
            for table in table_config['tables']
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            table_name = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"Successfully processed table: {table_name}")
                else:
                    print(f"Failed to process table: {table_name}")
            except Exception as e:
                print(f"Error processing table {table_name}: {str(e)}")
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 
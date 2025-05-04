from pyspark.sql import SparkSession
from utils.helpers import create_delta_table_if_not_exists, load_config

def create_spark_session():
    """Create and configure Spark session."""
    return SparkSession.builder \
        .appName("DataQualityAnomalyDetection") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

def main():
    # Initialize Spark session
    spark = create_spark_session()
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Create profiling results table
    profiling_schema = """
    (
        table_name STRING,
        database STRING,
        time_grain STRING,
        profiling_date TIMESTAMP,
        profile_data STRING
    )
    """
    
    create_delta_table_if_not_exists(
        spark,
        config['delta']['catalog'],
        config['delta']['schema'],
        config['delta']['profiling_table'],
        profiling_schema
    )
    
    # Create anomaly detection results table
    anomaly_schema = """
    (
        table_name STRING,
        database STRING,
        detection_date TIMESTAMP,
        anomaly_data STRING
    )
    """
    
    create_delta_table_if_not_exists(
        spark,
        config['delta']['catalog'],
        config['delta']['schema'],
        config['delta']['anomaly_table'],
        anomaly_schema
    )
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 
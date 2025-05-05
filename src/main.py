from pyspark.sql import SparkSession
from profiling.data_profiler import DataProfiler
from anomaly.anomaly_detector import AnomalyDetector
from anomaly.advanced_anomaly_detector import AdvancedAnomalyDetector
import yaml
import logging
from datetime import datetime

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_spark_session():
    """Create and configure Spark session."""
    return SparkSession.builder \
        .appName("GenericAnomalyDQ") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Setup
    logger = setup_logging()
    spark = create_spark_session()
    config = load_config()
    
    # Initialize components
    profiler = DataProfiler(spark)
    anomaly_detector = AnomalyDetector(spark)
    advanced_detector = AdvancedAnomalyDetector(spark)
    
    try:
        # Process each configured table
        for table_name in config['tables']:
            logger.info(f"Processing table: {table_name}")
            
            # Profile the table
            logger.info(f"Profiling {table_name}...")
            profile_results = profiler.profile_table(table_name)
            logger.info(f"Completed profiling for {table_name}")
            
            # Detect anomalies
            logger.info(f"Detecting anomalies for {table_name}...")
            anomalies = anomaly_detector.detect_anomalies(profile_results)
            logger.info(f"Found {len(anomalies)} basic anomalies for {table_name}")
            
            # Advanced anomaly detection
            logger.info(f"Running advanced anomaly detection for {table_name}...")
            advanced_anomalies = advanced_detector.detect_anomalies(profile_results)
            logger.info(f"Found {len(advanced_anomalies)} advanced anomalies for {table_name}")
            
            # Combine and save results
            all_anomalies = {
                'basic': anomalies,
                'advanced': advanced_anomalies
            }
            anomaly_detector.save_anomalies(all_anomalies, table_name)
            
            logger.info(f"Completed processing for {table_name}")
        
        logger.info("All tables processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing tables: {str(e)}")
        raise
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 
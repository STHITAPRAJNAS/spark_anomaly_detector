from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_timestamp
from typing import List, Dict, Any
import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_delta_table_if_not_exists(
    spark: SparkSession,
    catalog: str,
    schema: str,
    table: str,
    schema_str: str
):
    """Create a Delta table if it doesn't exist."""
    full_table_name = f"{catalog}.{schema}.{table}"
    
    # Check if table exists
    if not spark.catalog.tableExists(full_table_name):
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {full_table_name}
            USING DELTA
            {schema_str}
        """)

def get_latest_profiling_date(
    spark: SparkSession,
    catalog: str,
    schema: str,
    table: str,
    time_grain: str
) -> str:
    """Get the latest profiling date for a table."""
    full_table_name = f"{catalog}.{schema}.{table}"
    
    result = spark.sql(f"""
        SELECT MAX({time_grain}) as latest_date
        FROM {full_table_name}
    """).collect()[0]['latest_date']
    
    return result if result else None

def validate_table_exists(spark: SparkSession, database: str, table: str) -> bool:
    """Validate if a table exists in the specified database."""
    return spark.catalog.tableExists(f"{database}.{table}") 
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when, count, countDistinct, mean, stddev, min, max, approx_percentile, length, regexp_replace, trim, dayofweek
from pyspark.sql.types import StringType, NumericType, DateType, TimestampType
from typing import List, Dict, Any
import yaml
import json
from datetime import datetime

class DataProfiler:
    def __init__(self, spark: SparkSession, config_path: str = "config/config.yaml"):
        self.spark = spark
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _get_table_config(self, table_name: str) -> Dict[str, Any]:
        """Get configuration for a specific table."""
        if table_name not in self.config['tables']:
            raise ValueError(f"Table '{table_name}' not found in configuration")
        return self.config['tables'][table_name]
    
    def _get_global_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Get a global setting with fallback to default value."""
        return self.config['settings'].get(section, {}).get(key, default)
    
    def _apply_field_standardization(self, df: DataFrame, table_config: Dict[str, Any]) -> DataFrame:
        """Apply SQL expressions to standardize fields before profiling."""
        select_exprs = []
        
        # Add standardized columns from configuration
        for field_name, field_config in table_config['fields'].items():
            if 'standardization' in field_config:
                select_exprs.append(f"{field_config['standardization']} as {field_name}")
            else:
                select_exprs.append(f"{field_config['source_column']} as {field_name}")
        
        # Apply the transformations
        return df.selectExpr(*select_exprs)
    
    def _calculate_field_metrics(self, df: DataFrame, field_name: str, field_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for a specific field based on its configuration."""
        metrics = {}
        
        # Basic metrics available for all types
        if 'null_count' in field_config['metrics']:
            metrics['null_count'] = df.filter(col(field_name).isNull()).count()
            metrics['null_percentage'] = (metrics['null_count'] / df.count()) * 100
        
        # Type-specific metrics
        if isinstance(df.schema[field_name].dataType, NumericType):
            if 'min' in field_config['metrics']:
                metrics['min'] = df.select(min(col(field_name))).collect()[0][0]
            if 'max' in field_config['metrics']:
                metrics['max'] = df.select(max(col(field_name))).collect()[0][0]
            if 'mean' in field_config['metrics']:
                metrics['mean'] = df.select(mean(col(field_name))).collect()[0][0]
            if 'stddev' in field_config['metrics']:
                metrics['stddev'] = df.select(stddev(col(field_name))).collect()[0][0]
            if 'percentiles' in field_config['metrics']:
                percentiles = field_config.get('percentiles', [25, 50, 75])
                metrics['percentiles'] = df.select(
                    approx_percentile(col(field_name), percentiles, 0.01)
                ).collect()[0][0]
        
        elif isinstance(df.schema[field_name].dataType, StringType):
            if 'distinct_count' in field_config['metrics']:
                metrics['distinct_count'] = df.select(countDistinct(col(field_name))).collect()[0][0]
            if 'empty_count' in field_config['metrics']:
                metrics['empty_count'] = df.filter(col(field_name) == '').count()
            if field_config.get('pattern_analysis', False):
                metrics['pattern_analysis'] = {
                    'numeric_count': df.filter(col(field_name).rlike('^[0-9]+$')).count(),
                    'alpha_count': df.filter(col(field_name).rlike('^[a-zA-Z]+$')).count(),
                    'alphanumeric_count': df.filter(col(field_name).rlike('^[a-zA-Z0-9]+$')).count()
                }
        
        elif isinstance(df.schema[field_name].dataType, (DateType, TimestampType)):
            if 'min' in field_config['metrics']:
                metrics['min_date'] = str(df.select(min(col(field_name))).collect()[0][0])
            if 'max' in field_config['metrics']:
                metrics['max_date'] = str(df.select(max(col(field_name))).collect()[0][0])
            if 'day_of_week_distribution' in field_config['metrics']:
                day_dist = df.select(dayofweek(col(field_name)).alias('day')).groupBy('day').count()
                metrics['day_of_week_distribution'] = {row['day']: row['count'] for row in day_dist.collect()}
        
        return metrics
    
    def _calculate_custom_metrics(self, df: DataFrame, table_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate custom metrics defined in the configuration."""
        custom_metrics = {}
        
        for metric in table_config.get('custom_metrics', []):
            result = df.selectExpr(metric['expression']).collect()[0][0]
            custom_metrics[metric['name']] = {
                'value': result,
                'description': metric['description']
            }
        
        return custom_metrics
    
    def profile_table(self, table_name: str) -> Dict[str, Any]:
        """Profile a table using its specific configuration."""
        # Get table configuration
        table_config = self._get_table_config(table_name)
        database = table_config['database']
        
        # Read the table
        df = self.spark.table(f"{database}.{table_name}")
        
        # Apply field standardization
        df = self._apply_field_standardization(df, table_config)
        
        # Get sample size (table-specific or default)
        sample_size = table_config.get('sample_size', 
            self._get_global_setting('profiling', 'default_sample_size', 0.1))
        
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            df = df.sample(sample_size)
        elif isinstance(sample_size, int) and sample_size > 0:
            df = df.limit(sample_size)
        
        # Initialize results
        results = {
            'table_name': table_name,
            'database': database,
            'profile_date': datetime.now().isoformat(),
            'row_count': df.count(),
            'column_count': len(table_config['fields']),
            'fields': {}
        }
        
        # Profile each configured field
        for field_name, field_config in table_config['fields'].items():
            results['fields'][field_name] = {
                'source_column': field_config['source_column'],
                'metrics': self._calculate_field_metrics(df, field_name, field_config)
            }
        
        # Add custom metrics
        results['custom_metrics'] = self._calculate_custom_metrics(df, table_config)
        
        # Save results
        self._save_profile_results(results)
        
        return results
    
    def _save_profile_results(self, results: Dict[str, Any]):
        """Save profiling results to Delta table."""
        catalog = self._get_global_setting('storage', 'catalog')
        schema = self._get_global_setting('storage', 'schema')
        table = self._get_global_setting('storage', 'tables')['profiles']
        
        # Convert results to DataFrame with a single row
        results_df = self.spark.createDataFrame([(
            results['table_name'],
            results['database'],
            results['profile_date'],
            results['row_count'],
            results['column_count'],
            json.dumps(results['fields']),
            json.dumps(results['custom_metrics'])
        )], ['table_name', 'database', 'profile_date', 'row_count', 'column_count', 'field_stats', 'custom_metrics'])
        
        # Write to Delta table
        results_df.write.format("delta").mode("append").saveAsTable(
            f"{catalog}.{schema}.{table}"
        ) 
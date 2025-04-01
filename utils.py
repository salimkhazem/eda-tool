import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

def export_results(df, metadata, eda_plan, results, output_dir, formats=None):
    """
    Export EDA results in various formats
    
    Args:
        df: Pandas DataFrame with the dataset
        metadata: Dictionary with dataset metadata
        eda_plan: Dictionary with the EDA plan from GPT
        results: Dictionary with EDA results
        output_dir: Directory to save exports
        formats: List of export formats ('json', 'csv', 'excel')
        
    Returns:
        dict: Dictionary with paths to exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if formats is None:
        formats = ['json']
    
    export_paths = {}
    
    # Always export the raw analysis results as JSON
    results_path = output_dir / "eda_results.json"
    with open(results_path, 'w') as f:
        # Combine all results into a single JSON
        combined_results = {
            "metadata": metadata,
            "eda_plan": eda_plan,
            "results": results
        }
        
        # Handle non-serializable items
        json.dump(combined_results, f, indent=2, default=str)
    
    export_paths['json'] = str(results_path)
    
    # Export dataset summary
    if 'csv' in formats:
        # Create a summary DataFrame
        summary_data = []
        
        # Add column stats
        for col, col_type in metadata['column_dtypes'].items():
            col_data = {
                'column_name': col,
                'data_type': col_type,
                'classification': eda_plan.get('column_classifications', {}).get(col, 'Unknown'),
                'missing_values': metadata['missing_values'].get(col, 0),
                'missing_percentage': metadata['missing_values'].get(col, 0) / metadata['num_rows'] * 100 if metadata['num_rows'] > 0 else 0
            }
            
            # Add additional statistics based on column type
            if col in results['statistics']:
                for stat, value in results['statistics'][col].items():
                    col_data[f'stat_{stat}'] = value
            
            summary_data.append(col_data)
        
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / "eda_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        export_paths['csv'] = str(summary_path)
    
    # Export to Excel with multiple sheets
    if 'excel' in formats:
        excel_path = output_dir / "eda_complete.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            # Data sample
            df.head(20).to_excel(writer, sheet_name='Data Sample', index=False)
            
            # Column summary
            if 'csv' in formats:
                summary_df.to_excel(writer, sheet_name='Column Summary', index=False)
            
            # Statistics per column type
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            categorical_cols = [col for col in df.columns if col not in numeric_cols]
            
            if numeric_cols:
                df[numeric_cols].describe().T.to_excel(writer, sheet_name='Numeric Stats')
            
            if categorical_cols:
                cat_stats = pd.DataFrame({
                    'unique_values': [df[col].nunique() for col in categorical_cols],
                    'top_value': [df[col].value_counts().index[0] if not df[col].value_counts().empty else None for col in categorical_cols],
                    'top_count': [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in categorical_cols],
                    'missing_count': [df[col].isna().sum() for col in categorical_cols]
                }, index=categorical_cols)
                cat_stats.to_excel(writer, sheet_name='Categorical Stats')
            
            # Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                df[numeric_cols].corr().to_excel(writer, sheet_name='Correlations')
        
        export_paths['excel'] = str(excel_path)
    
    return export_paths

def detect_anomalies(df, column, method='iqr', threshold=1.5):
    """
    Detect anomalies in a numeric column
    
    Args:
        df: Pandas DataFrame
        column: Column name to analyze
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for anomaly detection
        
    Returns:
        DataFrame: Filtered DataFrame with only anomaly rows
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    if method.lower() == 'iqr':
        # IQR method
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    elif method.lower() == 'zscore':
        # Z-score method
        mean = df[column].mean()
        std = df[column].std()
        
        if std == 0:
            return pd.DataFrame()  # No variation, no anomalies
            
        z_scores = abs((df[column] - mean) / std)
        return df[z_scores > threshold]
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def create_derived_features(df):
    """
    Create common derived features based on data types
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        DataFrame: DataFrame with additional derived features
    """
    df_new = df.copy()
    features_added = []
    
    # Process datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        # Extract date components
        df_new[f'{col}_year'] = df[col].dt.year
        df_new[f'{col}_month'] = df[col].dt.month
        df_new[f'{col}_day'] = df[col].dt.day
        df_new[f'{col}_dayofweek'] = df[col].dt.dayofweek
        
        features_added.extend([
            f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'
        ])
    
    # For numeric columns, create squared and log features
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Only process columns with positive values for log
        if df[col].min() > 0:
            df_new[f'{col}_log'] = np.log(df[col])
            features_added.append(f'{col}_log')
        
        # Square for columns that aren't too large
        if df[col].max() < 1000:
            df_new[f'{col}_squared'] = df[col] ** 2
            features_added.append(f'{col}_squared')
    
    # For categorical columns with low cardinality, create one-hot encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Only for low cardinality
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_new = pd.concat([df_new, one_hot], axis=1)
            features_added.extend(one_hot.columns.tolist())
    
    return df_new, features_added

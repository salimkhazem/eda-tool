import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from datetime import datetime

class DataLoader:
    """Handles loading and basic inspection of data from various sources."""
    
    def __init__(self, config):
        self.config = config
        
    def load_data(self, data_source):
        """
        Load data from CSV, Excel file, or Pandas DataFrame
        
        Args:
            data_source: Path to CSV/Excel file or Pandas DataFrame
            
        Returns:
            DataFrame: Loaded data
        """
        if isinstance(data_source, pd.DataFrame):
            return data_source
        
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {data_source}")
            
        if path.suffix.lower() == '.csv':
            # Try to infer datetime columns and other settings
            try:
                return pd.read_csv(path, parse_dates=True, infer_datetime_format=True)
            except:
                # Fall back to default if parsing dates fails
                return pd.read_csv(path)
                
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                return pd.read_excel(path, parse_dates=True)
            except:
                return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def extract_metadata(self, df):
        """
        Extract metadata from the DataFrame
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            dict: Metadata dictionary
        """
        # Get basic info
        metadata = {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "column_dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample_data": df.head(self.config.max_rows_sample).to_dict(orient="records"),
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
            "column_descriptions": {}
        }
        
        # Add basic column descriptions with improved type detection
        for col in df.columns:
            # For numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it might be categorical despite numeric dtype
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    # Likely a categorical variable stored as numeric
                    metadata["column_descriptions"][col] = {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "unique_values": int(df[col].nunique()),
                        "suggested_type": "categorical_numeric"
                    }
                else:
                    # Regular numeric
                    metadata["column_descriptions"][col] = {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                        "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                        "skew": float(df[col].skew()) if hasattr(df[col], 'skew') and not pd.isna(df[col].skew()) else None,
                        "suggested_type": "numerical"
                    }
            
            # For string/object columns
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                unique_vals = df[col].nunique()
                unique_ratio = unique_vals / len(df) if len(df) > 0 else 1
                
                # Check if it might be a datetime
                date_count = 0
                for val in df[col].dropna().head(10).astype(str):
                    # Check common date patterns
                    if re.match(r'\d{4}-\d{2}-\d{2}', str(val)) or \
                       re.match(r'\d{2}/\d{2}/\d{4}', str(val)) or \
                       re.match(r'\d{2}-\d{2}-\d{4}', str(val)):
                        date_count += 1
                
                date_ratio = date_count / min(10, df[col].dropna().shape[0]) if df[col].dropna().shape[0] > 0 else 0
                
                if date_ratio > 0.7:
                    # Likely a datetime column
                    metadata["column_descriptions"][col] = {
                        "unique_values": int(unique_vals),
                        "suggested_type": "datetime"
                    }
                elif unique_ratio > 0.9 and unique_vals > 100:
                    # Likely an ID or text field
                    metadata["column_descriptions"][col] = {
                        "unique_values": int(unique_vals),
                        "avg_length": float(df[col].astype(str).str.len().mean()),
                        "suggested_type": "text" if df[col].astype(str).str.len().mean() > 20 else "id"
                    }
                else:
                    # Categorical
                    metadata["column_descriptions"][col] = {
                        "unique_values": int(unique_vals),
                        "is_categorical": True,
                        "top_5_values": df[col].value_counts().head(5).to_dict(),
                        "suggested_type": "categorical"
                    }
            
            # For datetime columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                if df[col].dropna().shape[0] > 0:
                    metadata["column_descriptions"][col] = {
                        "min": df[col].min().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[col].min()) else None,
                        "max": df[col].max().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[col].max()) else None,
                        "suggested_type": "datetime"
                    }
                else:
                    metadata["column_descriptions"][col] = {
                        "suggested_type": "datetime",
                        "note": "Empty datetime column"
                    }
            
            # For boolean columns
            elif pd.api.types.is_bool_dtype(df[col]):
                metadata["column_descriptions"][col] = {
                    "true_count": int(df[col].sum()),
                    "false_count": int(len(df) - df[col].sum()),
                    "true_percentage": float(df[col].mean() * 100),
                    "suggested_type": "boolean"
                }
        
        return metadata
    
    def preprocess_data(self, df):
        """
        Apply basic preprocessing steps to the DataFrame
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            DataFrame: Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Convert object columns that look like dates to datetime
        for col in df_processed.select_dtypes(include=['object']):
            try:
                # Look at a sample to check if it's a date
                sample = df_processed[col].dropna().head(5)
                date_count = 0
                
                for val in sample:
                    if re.match(r'\d{4}-\d{2}-\d{2}', str(val)) or \
                       re.match(r'\d{2}/\d{2}/\d{4}', str(val)) or \
                       re.match(r'\d{2}-\d{2}-\d{4}', str(val)):
                        date_count += 1
                
                # If most of the samples look like dates, convert the column
                if date_count >= 3:  # 60% of samples
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    print(f"Converted column '{col}' to datetime")
            except:
                # If conversion fails, keep as is
                pass
        
        # Fill missing values for numerical columns with median
        numeric_cols = df_processed.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            missing_pct = df_processed[col].isna().mean()
            if 0 < missing_pct < 0.2:  # Only fill if < 20% missing
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                print(f"Filled missing values in '{col}' with median")
                
        return df_processed

import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from openai import AzureOpenAI

class OpenAIService:
    """Handles interactions with Azure OpenAI API."""
    
    def __init__(self, config):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint
        )
    
    def generate_eda_prompt(self, metadata):
        """
        Generate a prompt for EDA based on data metadata
        
        Args:
            metadata: Dictionary containing dataset metadata
            
        Returns:
            str: Prompt for GPT model
        """
        # Create a custom JSON encoder to handle Pandas Timestamp objects
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Format sample data for the prompt
        sample_data_str = json.dumps(metadata['sample_data'], indent=2, cls=CustomJSONEncoder)
        
        prompt = f"""
You are a data science expert specialized in Exploratory Data Analysis (EDA).
I have a dataset with {metadata['num_rows']} rows and {metadata['num_columns']} columns.

The column names and data types are:
{json.dumps({c: t for c, t in metadata['column_dtypes'].items()}, indent=2)}

Here are the first {len(metadata['sample_data'])} rows of the dataset:
{sample_data_str}

Missing values per column:
{json.dumps(metadata['missing_values'], indent=2)}

Based on this information, please provide a COMPLETE Exploratory Data Analysis plan in JSON format following exactly this structure:
```json
{{
  "column_classifications": {{
    "column_name": "classification_type"  // numerical, categorical, datetime, text, or other
  }},
  "statistics": [
    {{
      "column": "column_name",
      "stats": ["stat1", "stat2"]  // e.g., mean, median, mode, std, min, max, etc.
    }}
  ],
  "visualizations": [
    {{
      "title": "Plot Title",
      "type": "plot_type",  // e.g., histogram, scatter, bar, box, correlation, pie, line, pairplot, density
      "columns": ["column1", "column2"],
      "description": "What this visualization shows"
    }}
  ],
  "data_quality_checks": [
    {{
      "check_type": "check_name",  // e.g., missing_values, duplicates, outliers
      "columns": ["column1"],
      "description": "What this check evaluates"
    }}
  ],
  "feature_engineering": [
    {{
      "name": "new_feature_name",
      "description": "How to create this feature",
      "columns_used": ["column1", "column2"]
    }}
  ],
  "insights": [
    "Potential insight 1",
    "Potential insight 2"
  ]
}}
```

Supported visualization types include: histogram, scatter, bar, box/boxplot, correlation/heatmap, pie, line, pairplot, and density/kde.
Choose the most appropriate visualizations based on the data types:
- For numerical columns: histogram, boxplot, density, scatter (with other numerical columns)
- For categorical columns: bar charts, pie charts
- For datetime columns: line charts with numerical values
- For relationships: scatter plots, correlation heatmaps, box plots of numerical by categorical

The depth level for this analysis is: {self.config.eda_depth}. Please tailor the complexity accordingly:
- basic: Focus on univariate analysis and simple checks
- intermediate: Include bivariate analysis and more detailed statistics
- deep: Comprehensive analysis with multivariate visualizations and advanced feature engineering

Your response must be a valid JSON object that I can parse. Include only the JSON in your response, no additional text.
"""
        return prompt
        
    def get_eda_plan(self, metadata):
        """
        Request an EDA plan from GPT model
        
        Args:
            metadata: Dictionary containing dataset metadata
            
        Returns:
            dict: Parsed JSON response with EDA plan
        """
        prompt = self.generate_eda_prompt(metadata)
        
        response = self.client.chat.completions.create(
            model=self.config.azure_deployment,
            messages=[
                {"role": "system", "content": "You are a data science expert that creates comprehensive Exploratory Data Analysis plans in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
            raise

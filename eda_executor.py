import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from datetime import datetime

class EDAExecutor:
    """Executes the EDA plan suggested by GPT."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.results = {
            "statistics": {},
            "visualizations": [],
            "data_quality": {},
            "feature_engineering": {},
            "insights": [],
            "deep_analysis": {}
        }
        
    def execute_plan(self, df, eda_plan, perform_deep_analysis=False):
        """
        Execute the EDA plan on the dataframe
        
        Args:
            df: Pandas DataFrame
            eda_plan: Dictionary containing the EDA plan from GPT
            perform_deep_analysis: Whether to perform deep analysis
            
        Returns:
            dict: Results of the execution
        """
        # Apply column classifications
        classifications = eda_plan.get("column_classifications", {})
        
        # Calculate statistics
        self._calculate_statistics(df, eda_plan.get("statistics", []))
        
        # Create visualizations
        if self.config.eda_depth == 'deep':
            # For deep analysis, use the visualization engine for better plots
            try:
                from visualization_engine import VisualizationEngine
                viz_engine = VisualizationEngine(self.config)
                viz_results = viz_engine.create_all_visualizations(df, eda_plan)
                self.results["visualizations"] = viz_results
                
                # Create visualization dashboard
                dashboard_path = viz_engine.get_visualization_dashboard()
                self.results["visualization_dashboard"] = dashboard_path
                
                print(f"   ✓ Created visualizations dashboard at: {dashboard_path}")
            except ImportError:
                print("   ⚠️ Visualization engine not available, using basic visualizations")
                self._create_visualizations(df, eda_plan.get("visualizations", []))
        else:
            # Use traditional visualization approach for basic/intermediate depths
            self._create_visualizations(df, eda_plan.get("visualizations", []))
        
        # Perform data quality checks
        self._check_data_quality(df, eda_plan.get("data_quality_checks", []))
        
        # Generate feature engineering examples
        self._engineer_features(df, eda_plan.get("feature_engineering", []))
        
        # Store insights
        self.results["insights"] = eda_plan.get("insights", [])
        
        # Perform deep analysis if requested and if depth is set to deep
        if perform_deep_analysis or self.config.eda_depth == 'deep':
            try:
                print("🔬 Performing deep analysis...")
                from deep_eda import DeepEDA
                deep_analyzer = DeepEDA(self.config)
                deep_results = deep_analyzer.perform_deep_analysis(df, self._get_metadata(df), eda_plan)
                self.results["deep_analysis"] = deep_results
                print(f"   ✓ Deep analysis completed with {len(deep_results)} components")
            except ImportError:
                print("   ⚠️ Deep analysis module not available, skipping")
        
        return self.results
    
    def _get_metadata(self, df):
        """
        Generate basic metadata for the dataframe (used for deep analysis)
        """
        return {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "column_dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
    
    def _calculate_statistics(self, df, stats_list):
        """Calculate requested statistics for columns"""
        for stat_item in stats_list:
            column = stat_item.get("column")
            stats = stat_item.get("stats", [])
            
            if column not in df.columns:
                continue
                
            self.results["statistics"][column] = {}
            
            for stat in stats:
                if stat.lower() == "mean":
                    self.results["statistics"][column]["mean"] = float(df[column].mean()) if pd.api.types.is_numeric_dtype(df[column]) else None
                elif stat.lower() == "median":
                    self.results["statistics"][column]["median"] = float(df[column].median()) if pd.api.types.is_numeric_dtype(df[column]) else None
                elif stat.lower() == "mode":
                    self.results["statistics"][column]["mode"] = df[column].mode()[0] if not df[column].mode().empty else None
                elif stat.lower() == "std":
                    self.results["statistics"][column]["std"] = float(df[column].std()) if pd.api.types.is_numeric_dtype(df[column]) else None
                elif stat.lower() == "min":
                    self.results["statistics"][column]["min"] = float(df[column].min()) if pd.api.types.is_numeric_dtype(df[column]) else None
                elif stat.lower() == "max":
                    self.results["statistics"][column]["max"] = float(df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else None
                elif stat.lower() == "count":
                    self.results["statistics"][column]["count"] = int(df[column].count())
                elif stat.lower() == "nunique":
                    self.results["statistics"][column]["nunique"] = int(df[column].nunique())
    
    def _create_visualizations(self, df, viz_list):
        """Create requested visualizations"""
        for i, viz in enumerate(viz_list):
            title = viz.get("title", f"Plot {i+1}")
            plot_type = viz.get("type", "histogram")
            columns = viz.get("columns", [])
            description = viz.get("description", "")
            
            # Skip if required columns don't exist
            if not all(col in df.columns for col in columns):
                continue
                
            fig_path = f"plot_{i+1}_{plot_type}.png"
            full_path = self.figures_dir / fig_path
            
            plt.figure(figsize=(10, 6))
            
            try:
                if plot_type.lower() == "histogram":
                    if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                        sns.histplot(df[columns[0]].dropna(), kde=True)
                        plt.xlabel(columns[0])
                        plt.ylabel("Frequency")
                    
                elif plot_type.lower() == "scatter":
                    if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                        if len(columns) >= 3:  # Use third column for hue if available
                            sns.scatterplot(x=df[columns[0]], y=df[columns[1]], hue=df[columns[2]])
                            plt.legend(title=columns[2], bbox_to_anchor=(1.05, 1), loc='upper left')
                        else:
                            sns.scatterplot(x=df[columns[0]], y=df[columns[1]])
                        plt.xlabel(columns[0])
                        plt.ylabel(columns[1])
                    
                elif plot_type.lower() == "bar":
                    if len(columns) == 1:
                        value_counts = df[columns[0]].value_counts().sort_values(ascending=False).head(10)
                        value_counts.plot(kind='bar')
                        plt.xlabel(columns[0])
                        plt.ylabel("Count")
                        plt.xticks(rotation=45, ha='right')
                    elif len(columns) == 2:  # Grouped bar chart
                        pd.crosstab(df[columns[0]], df[columns[1]]).plot(kind='bar', stacked=False)
                        plt.xlabel(columns[0])
                        plt.ylabel("Count")
                        plt.xticks(rotation=45, ha='right')
                        plt.legend(title=columns[1], bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                elif plot_type.lower() == "box" or plot_type.lower() == "boxplot":
                    if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                        sns.boxplot(y=df[columns[0]])
                        plt.ylabel(columns[0])
                    elif len(columns) == 2:  # Box plot with categorical x-axis
                        if pd.api.types.is_numeric_dtype(df[columns[0]]) and not pd.api.types.is_numeric_dtype(df[columns[1]]):
                            sns.boxplot(x=df[columns[1]], y=df[columns[0]])
                            plt.xlabel(columns[1])
                            plt.ylabel(columns[0])
                            plt.xticks(rotation=45, ha='right')
                        elif pd.api.types.is_numeric_dtype(df[columns[1]]) and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                            sns.boxplot(x=df[columns[0]], y=df[columns[1]])
                            plt.xlabel(columns[0])
                            plt.ylabel(columns[1])
                            plt.xticks(rotation=45, ha='right')
                    
                elif plot_type.lower() == "correlation" or plot_type.lower() == "heatmap":
                    corr_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
                    if corr_cols:
                        corr_matrix = df[corr_cols].corr()
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                                    mask=mask, vmin=-1, vmax=1)
                        plt.tight_layout()
                
                elif plot_type.lower() == "pie":
                    if len(columns) == 1 and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                        counts = df[columns[0]].value_counts()
                        # Limit to top categories if too many
                        if len(counts) > 8:
                            other_count = counts[7:].sum()
                            counts = counts[:7]
                            counts['Other'] = other_count
                        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                        plt.axis('equal')
                
                elif plot_type.lower() == "line":
                    if len(columns) >= 2:
                        # Check if first column is datetime
                        if pd.api.types.is_datetime64_any_dtype(df[columns[0]]) or isinstance(df[columns[0]].iloc[0], (datetime, str)):
                            # Try to convert to datetime if it's a string
                            if isinstance(df[columns[0]].iloc[0], str):
                                try:
                                    df[columns[0]] = pd.to_datetime(df[columns[0]])
                                except:
                                    pass
                            
                            # Plot each numeric column as a line
                            for col in columns[1:]:
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    plt.plot(df[columns[0]], df[col], label=col)
                            
                            plt.xlabel(columns[0])
                            plt.ylabel("Value")
                            plt.legend()
                            plt.xticks(rotation=45, ha='right')
                
                elif plot_type.lower() == "pairplot" and len(columns) > 1:
                    # Use a subset of columns if there are many
                    plot_cols = columns[:5] if len(columns) > 5 else columns
                    sns.pairplot(df[plot_cols])
                    plt.tight_layout()
                
                elif plot_type.lower() == "density" or plot_type.lower() == "kde":
                    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
                    if numeric_cols:
                        for col in numeric_cols:
                            sns.kdeplot(df[col].dropna(), label=col)
                        plt.xlabel("Value")
                        plt.ylabel("Density")
                        plt.legend()
                
                plt.title(title)
                plt.tight_layout()
                plt.savefig(full_path)
                plt.close()
                
                self.results["visualizations"].append({
                    "title": title,
                    "description": description,
                    "file_path": str(full_path),
                    "type": plot_type,
                    "columns": columns
                })
                
            except Exception as e:
                print(f"Error creating {plot_type} plot for {columns}: {e}")
    
    def _check_data_quality(self, df, checks):
        """Perform data quality checks"""
        for check in checks:
            check_type = check.get("check_type", "")
            columns = check.get("columns", [])
            description = check.get("description", "")
            
            if not all(col in df.columns for col in columns):
                continue
                
            check_id = f"{check_type}_{'-'.join(columns)}"
            self.results["data_quality"][check_id] = {
                "type": check_type,
                "description": description,
                "columns": columns,
                "results": {}
            }
            
            try:
                if check_type.lower() == "missing_values":
                    for col in columns:
                        missing = df[col].isna().sum()
                        missing_pct = missing / len(df) * 100
                        self.results["data_quality"][check_id]["results"][col] = {
                            "missing_count": int(missing),
                            "missing_percentage": float(missing_pct)
                        }
                
                elif check_type.lower() == "duplicates":
                    dups = df.duplicated(subset=columns).sum()
                    self.results["data_quality"][check_id]["results"] = {
                        "duplicate_count": int(dups),
                        "duplicate_percentage": float(dups / len(df) * 100)
                    }
                
                elif check_type.lower() == "outliers":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
                            
                            self.results["data_quality"][check_id]["results"][col] = {
                                "outlier_count": int(outliers),
                                "outlier_percentage": float(outliers / len(df) * 100),
                                "lower_bound": float(lower_bound),
                                "upper_bound": float(upper_bound)
                            }
            except Exception as e:
                print(f"Error performing {check_type} check: {e}")
    
    def _engineer_features(self, df, features):
        """Generate example engineered features"""
        for feature in features:
            name = feature.get("name", "")
            description = feature.get("description", "")
            columns_used = feature.get("columns_used", [])
            
            if not all(col in df.columns for col in columns_used):
                continue
                
            self.results["feature_engineering"][name] = {
                "description": description,
                "columns_used": columns_used,
                "example": "See report for implementation details"
            }

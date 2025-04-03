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
                
                # Automatically generate additional visualizations by enhancing the plan
                enhanced_plan = self._enhance_visualization_plan(df, eda_plan)
                
                viz_results = viz_engine.create_all_visualizations(df, enhanced_plan)
                self.results["visualizations"] = viz_results
                
                # Create visualization dashboard
                dashboard_path = viz_engine.get_visualization_dashboard()
                self.results["visualization_dashboard"] = dashboard_path
                
                print(f"   ✓ Created {len(viz_results)} visualizations")
                print(f"   ✓ Created visualizations dashboard at: {dashboard_path}")
            except ImportError:
                print("   ⚠️ Visualization engine not available, using basic visualizations")
                self._create_visualizations(df, eda_plan.get("visualizations", []))
        else:
            # Use traditional visualization approach for basic/intermediate depths
            enhanced_plan = self._enhance_visualization_plan(df, eda_plan) if self.config.eda_depth == 'intermediate' else eda_plan
            self._create_visualizations(df, enhanced_plan.get("visualizations", []))
        
        # Perform data quality checks
        self._check_data_quality(df, eda_plan.get("data_quality_checks", []))
        
        # Generate feature engineering examples
        self._engineer_features(df, eda_plan.get("feature_engineering", []))
        
        # Store insights
        self.results["insights"] = eda_plan.get("insights", [])
        
        # Create enhanced correlation visualizations
        try:
            from correlation_viz import (
                create_enhanced_correlation_heatmap,
                create_clustered_correlation_heatmap,
                create_absolute_correlation_heatmap,
                create_filtered_correlation_heatmap
            )
            
            # Generate the Pearson correlation heatmap directly in the figures directory
            heatmap_path = self.figures_dir / "correlation_heatmap.png"
            corr_path = create_enhanced_correlation_heatmap(df, output_path=heatmap_path, method='pearson')
            
            # Add standard Pearson correlation visualization to results
            if corr_path:
                self.results["visualizations"].append({
                    "title": "Pearson Correlation Heatmap",
                    "type": "correlation",
                    "columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "description": "Heatmap showing Pearson correlations between numeric variables",
                    "file_path": str(corr_path)
                })
            
            # Generate the Spearman correlation heatmap directly in the figures directory
            spearman_path = self.figures_dir / "spearman_correlation_heatmap.png"
            spearman_corr_path = create_enhanced_correlation_heatmap(df, output_path=spearman_path, method='spearman')
            
            # Add Spearman correlation visualization to results
            if spearman_corr_path:
                self.results["visualizations"].append({
                    "title": "Spearman Correlation Heatmap",
                    "type": "correlation",
                    "columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "description": "Heatmap showing Spearman rank correlations between numeric variables",
                    "file_path": str(spearman_corr_path)
                })
            
            # Add clustered correlation heatmap
            clustered_path = self.figures_dir / "clustered_correlation_heatmap.png"
            clustered_corr_path = create_clustered_correlation_heatmap(df, output_path=clustered_path, method='pearson')
            if clustered_corr_path:
                self.results["visualizations"].append({
                    "title": "Clustered Correlation Heatmap",
                    "type": "correlation",
                    "columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "description": "Hierarchically clustered heatmap showing Pearson correlations between numeric variables",
                    "file_path": str(clustered_corr_path)
                })
            
            # Add clustered Spearman correlation heatmap
            clustered_spearman_path = self.figures_dir / "clustered_spearman_correlation_heatmap.png"
            clustered_spearman_corr_path = create_clustered_correlation_heatmap(df, output_path=clustered_spearman_path, method='spearman')
            if clustered_spearman_corr_path:
                self.results["visualizations"].append({
                    "title": "Clustered Spearman Correlation Heatmap", 
                    "type": "correlation",
                    "columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "description": "Hierarchically clustered heatmap showing Spearman rank correlations between numeric variables",
                    "file_path": str(clustered_spearman_corr_path)
                })
            
            # Add absolute correlation heatmap
            absolute_path = self.figures_dir / "absolute_correlation_heatmap.png"
            abs_corr_path = create_absolute_correlation_heatmap(df, output_path=absolute_path)
            if abs_corr_path:
                self.results["visualizations"].append({
                    "title": "Absolute Correlation Heatmap",
                    "type": "correlation",
                    "columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "description": "Heatmap showing absolute correlation values between numeric variables",
                    "file_path": str(abs_corr_path)
                })
            
            # Add filtered Pearson correlation heatmap
            filtered_path = self.figures_dir / "filtered_correlation_heatmap.png"
            filtered_corr_path = create_filtered_correlation_heatmap(df, output_path=filtered_path)
            if filtered_corr_path:
                self.results["visualizations"].append({
                    "title": "Strong Pearson Correlations Heatmap",
                    "type": "correlation",
                    "columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "description": "Heatmap showing only strong Pearson correlations (|r| > 0.5) between numeric variables",
                    "file_path": str(filtered_corr_path)
                })
                
            # Add filtered Spearman correlation heatmap
            filtered_spearman_path = self.figures_dir / "filtered_spearman_correlation_heatmap.png"
            filtered_spearman_corr_path = create_filtered_correlation_heatmap(df, output_path=filtered_spearman_path, method='spearman')
            if filtered_spearman_corr_path:
                self.results["visualizations"].append({
                    "title": "Strong Spearman Correlations Heatmap",
                    "type": "correlation",
                    "columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "description": "Heatmap showing only strong Spearman rank correlations (|r| > 0.5) between numeric variables",
                    "file_path": str(filtered_spearman_corr_path)
                })
                
        except ImportError:
            print("   ⚠️ correlation_viz module not available, skipping enhanced correlation visualizations")
        except Exception as e:
            print(f"   ⚠️ Error creating correlation visualizations: {e}")
        
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
    
    def _enhance_visualization_plan(self, df, eda_plan):
        """
        Enhance the visualization plan to include more automatically generated visualizations
        
        Args:
            df: Pandas DataFrame
            eda_plan: Original EDA plan
            
        Returns:
            dict: Enhanced EDA plan
        """
        # Create a deep copy of the original plan
        import copy
        enhanced_plan = copy.deepcopy(eda_plan)
        original_viz_count = len(enhanced_plan.get("visualizations", []))
        
        # Get all numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # If plan doesn't have visualizations, create an empty list
        if "visualizations" not in enhanced_plan:
            enhanced_plan["visualizations"] = []
        
        existing_viz_types = {}
        # Track which visualizations already exist to avoid duplicates
        for viz in enhanced_plan["visualizations"]:
            viz_type = viz.get("type", "").lower()
            viz_columns = tuple(sorted(viz.get("columns", [])))
            key = f"{viz_type}_{viz_columns}"
            existing_viz_types[key] = True
        
        # Add distribution plots for each numeric column
        for col in numeric_columns[:10]:  # Limit to first 10 numeric columns
            key = f"histogram_{(col,)}"
            if key not in existing_viz_types:
                enhanced_plan["visualizations"].append({
                    "title": f"Distribution of {col}",
                    "type": "histogram",
                    "columns": [col],
                    "description": f"Histogram showing the distribution of {col}"
                })
        
        # Add box plots for each numeric column
        for col in numeric_columns[:10]:  # Limit to first 10 numeric columns
            key = f"box_{(col,)}"
            if key not in existing_viz_types:
                enhanced_plan["visualizations"].append({
                    "title": f"Box Plot of {col}",
                    "type": "box",
                    "columns": [col],
                    "description": f"Box plot showing the distribution and outliers of {col}"
                })
        
        # Add count plots for each categorical column
        for col in categorical_columns[:10]:  # Limit to first 10 categorical columns
            key = f"bar_{(col,)}"
            if key not in existing_viz_types:
                enhanced_plan["visualizations"].append({
                    "title": f"Count of {col}",
                    "type": "bar",
                    "columns": [col],
                    "description": f"Bar chart showing the frequency of each category in {col}"
                })
        
        # Add some scatter plots between numeric columns
        if len(numeric_columns) >= 2:
            for i in range(min(5, len(numeric_columns))):
                for j in range(i+1, min(5, len(numeric_columns))):
                    col1 = numeric_columns[i]
                    col2 = numeric_columns[j]
                    key = f"scatter_{tuple(sorted([col1, col2]))}"
                    if key not in existing_viz_types:
                        enhanced_plan["visualizations"].append({
                            "title": f"Scatter Plot of {col1} vs {col2}",
                            "type": "scatter",
                            "columns": [col1, col2],
                            "description": f"Scatter plot showing the relationship between {col1} and {col2}"
                        })
        
        # Add time series plots if there are datetime columns
        if datetime_columns and numeric_columns:
            for dt_col in datetime_columns[:2]:  # Limit to first 2 datetime columns
                for num_col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                    key = f"line_{tuple(sorted([dt_col, num_col]))}"
                    if key not in existing_viz_types:
                        enhanced_plan["visualizations"].append({
                            "title": f"{num_col} over {dt_col}",
                            "type": "line",
                            "columns": [dt_col, num_col],
                            "description": f"Line chart showing how {num_col} changes over {dt_col}"
                        })
        
        # Add box plots of numeric columns by categorical columns
        if numeric_columns and categorical_columns:
            for num_col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                for cat_col in categorical_columns[:3]:  # Limit to first 3 categorical columns
                    # Only add if the categorical column doesn't have too many values
                    if df[cat_col].nunique() <= 10:
                        key = f"boxplot_{tuple(sorted([num_col, cat_col]))}"
                        if key not in existing_viz_types:
                            enhanced_plan["visualizations"].append({
                                "title": f"{num_col} by {cat_col}",
                                "type": "boxplot",
                                "columns": [cat_col, num_col],  # Categorical first, then numeric
                                "description": f"Box plot showing distribution of {num_col} for each category in {cat_col}"
                            })
        
        # Add correlation heatmap for all numeric columns
        if len(numeric_columns) > 3:
            key = f"heatmap_{tuple(sorted(numeric_columns))}"
            if key not in existing_viz_types:
                enhanced_plan["visualizations"].append({
                    "title": "Correlation Heatmap",
                    "type": "heatmap",
                    "columns": numeric_columns,
                    "description": "Heatmap showing the correlation between numeric variables"
                })
        
        # Add pairplot for a subset of numeric columns
        if len(numeric_columns) >= 3:
            # If we have categorical columns, include one for coloring
            if categorical_columns and df[categorical_columns[0]].nunique() <= 5:
                paired_cols = numeric_columns[:4] + [categorical_columns[0]]
                key = f"pairplot_{tuple(sorted(paired_cols))}"
                if key not in existing_viz_types:
                    enhanced_plan["visualizations"].append({
                        "title": "Pair Plot with Categories",
                        "type": "pairplot",
                        "columns": paired_cols,
                        "description": "Pair plot showing relationships between numeric variables with categorical coloring"
                    })
            else:
                paired_cols = numeric_columns[:4]  # Limit to 4 columns for readability
                key = f"pairplot_{tuple(sorted(paired_cols))}"
                if key not in existing_viz_types:
                    enhanced_plan["visualizations"].append({
                        "title": "Pair Plot",
                        "type": "pairplot",
                        "columns": paired_cols,
                        "description": "Pair plot showing relationships between numeric variables"
                    })
        
        # For deep analysis, add specialized visualizations
        if self.config.eda_depth == 'deep':
            # Add violin plots for numeric columns by categorical
            if numeric_columns and categorical_columns:
                for num_col in numeric_columns[:2]:  # Limit to first 2 numeric columns
                    for cat_col in categorical_columns[:2]:  # Limit to first 2 categorical columns
                        if df[cat_col].nunique() <= 8:  # Only for smaller number of categories
                            key = f"violin_{tuple(sorted([num_col, cat_col]))}"
                            if key not in existing_viz_types:
                                enhanced_plan["visualizations"].append({
                                    "title": f"Violin Plot of {num_col} by {cat_col}",
                                    "type": "violin",
                                    "columns": [cat_col, num_col],
                                    "description": f"Violin plot showing distribution of {num_col} for each category in {cat_col}"
                                })
            
            # Add interactive plots
            if len(numeric_columns) >= 3:
                # 3D scatter plot
                key = f"3d_scatter_{tuple(sorted(numeric_columns[:3]))}"
                if key not in existing_viz_types:
                    enhanced_plan["visualizations"].append({
                        "title": f"3D Scatter Plot",
                        "type": "scatter3d",
                        "columns": numeric_columns[:3],
                        "description": "3D scatter plot showing relationships between three numeric variables",
                        "interactive": True
                    })
            
            # Add parallel coordinates plot
            if len(numeric_columns) >= 4:
                columns_to_use = numeric_columns[:6]  # Use up to 6 columns
                if categorical_columns:
                    columns_to_use.append(categorical_columns[0])  # Add a categorical column for color
                
                key = f"parallel_coordinates_{tuple(sorted(columns_to_use))}"
                if key not in existing_viz_types:
                    enhanced_plan["visualizations"].append({
                        "title": "Parallel Coordinates Plot",
                        "type": "parallel_coordinates",
                        "columns": columns_to_use,
                        "description": "Parallel coordinates plot showing relationships between multiple variables",
                        "interactive": True
                    })
        
        added_viz_count = len(enhanced_plan["visualizations"]) - original_viz_count
        print(f"   ✓ Added {added_viz_count} more visualizations to the plan")
        return enhanced_plan

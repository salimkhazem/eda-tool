import json
from datetime import datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os  # Add missing import for os module
import traceback  # Add for better error tracing

class ReportGenerator:
    """Generates EDA reports in different formats."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def generate_markdown(self, df, metadata, eda_plan, results):
        """
        Generate a markdown report of the EDA
        
        Args:
            df: Pandas DataFrame
            metadata: Dictionary with dataset metadata
            eda_plan: Dictionary with the EDA plan
            results: Dictionary with EDA results
            
        Returns:
            str: Path to the generated report
        """
        report_path = self.output_dir / "eda_report.md"
        
        with open(report_path, 'w') as f:
            # Title
            f.write(f"# Exploratory Data Analysis Report\n\n")
            
            # Dataset overview
            f.write(f"## Dataset Overview\n\n")
            f.write(f"- **Rows**: {metadata['num_rows']}\n")
            f.write(f"- **Columns**: {metadata['num_columns']}\n\n")
            
            # Column classifications
            f.write(f"## Column Classifications\n\n")
            f.write("| Column | Type | Classification |\n")
            f.write("|--------|------|----------------|\n")
            
            for col, dtype in metadata['column_dtypes'].items():
                classification = eda_plan.get("column_classifications", {}).get(col, "Unknown")
                f.write(f"| {col} | {dtype} | {classification} |\n")
            
            f.write("\n")
            
            # Data Sample
            f.write(f"## Data Sample\n\n")
            f.write(df.head(self.config.max_rows_sample).to_markdown())
            f.write("\n\n")
            
            # Statistics
            f.write(f"## Statistical Analysis\n\n")
            for col, stats in results['statistics'].items():
                f.write(f"### {col}\n\n")
                f.write("| Measure | Value |\n")
                f.write("|---------|-------|\n")
                for stat, value in stats.items():
                    f.write(f"| {stat} | {value} |\n")
                f.write("\n")
            
            # Visualizations
            f.write(f"## Visualizations\n\n")
            
            # Check if we have a visualization dashboard from the advanced visualization engine
            if "visualization_dashboard" in results:
                dashboard_path = Path(results["visualization_dashboard"]).name
                # Use absolute path for dashboard in the markdown
                f.write(f"An interactive visualization dashboard is available at: [{dashboard_path}]({dashboard_path})\n\n")
            
            # Include individual visualizations
            for viz in results['visualizations']:
                f.write(f"### {viz['title']}\n\n")
                f.write(f"**Type**: {viz['type']}\n\n")
                f.write(f"**Columns**: {', '.join(viz['columns'])}\n\n")
                f.write(f"**Description**: {viz['description']}\n\n")
                
                # For images, only use the filename without any path prefixes
                # This will work better with how we're processing images in Streamlit
                file_path = Path(viz['file_path'])
                img_filename = file_path.name
                
                f.write(f"![{viz['title']}]({img_filename})\n\n")
            
            # Data Quality
            f.write(f"## Data Quality Checks\n\n")
            for check_id, check in results['data_quality'].items():
                f.write(f"### {check['type']}\n\n")
                f.write(f"**Columns**: {', '.join(check['columns'])}\n\n")
                f.write(f"**Description**: {check['description']}\n\n")
                f.write("**Results**:\n\n")
                
                f.write("```\n")
                f.write(json.dumps(check['results'], indent=2))
                f.write("\n```\n\n")
            
            # Feature Engineering
            f.write(f"## Feature Engineering Ideas\n\n")
            for name, feature in results['feature_engineering'].items():
                f.write(f"### {name}\n\n")
                f.write(f"**Description**: {feature['description']}\n\n")
                f.write(f"**Columns used**: {', '.join(feature['columns_used'])}\n\n")
            
            # Deep Analysis results, if available
            if "deep_analysis" in results and results["deep_analysis"]:
                f.write(f"## Deep Analysis\n\n")
                
                # Comprehensive Pairplot (add this section)
                if "comprehensive_pairplot" in results["deep_analysis"] and "error" not in results["deep_analysis"]["comprehensive_pairplot"]:
                    f.write(f"### Comprehensive Pairplot\n\n")
                    pairplot_data = results["deep_analysis"]["comprehensive_pairplot"]
                    
                    f.write(f"This plot shows the pairwise relationships between key numeric variables in the dataset.\n\n")
                    
                    if pairplot_data["hue_variable"]:
                        f.write(f"Points are colored by: **{pairplot_data['hue_variable']}**\n\n")
                        
                    # Add the variables shown in the plot
                    f.write(f"**Variables included**: {', '.join(pairplot_data['variables'])}\n\n")
                    
                    # Include the plot image - use just the filename
                    if "plot_path" in pairplot_data:
                        file_path = Path(pairplot_data["plot_path"])
                        img_filename = file_path.name
                        f.write(f"![Comprehensive Pairplot]({img_filename})\n\n")
                        
                    f.write("The diagonal shows the distribution of each variable, while the off-diagonal cells show the relationship between pairs of variables.\n\n")
                
                # Distribution Analysis
                if "distribution_analysis" in results["deep_analysis"]:
                    f.write(f"### Distribution Analysis\n\n")
                    dist_results = results["deep_analysis"]["distribution_analysis"]
                    for col, dist_data in dist_results.items():
                        f.write(f"#### {col}\n\n")
                        
                        # Include basic stats
                        if "stats" in dist_data:
                            f.write("**Statistics**:\n\n")
                            f.write("| Measure | Value |\n")
                            f.write("|---------|-------|\n")
                            for stat, value in dist_data["stats"].items():
                                f.write(f"| {stat} | {value} |\n")
                            f.write("\n")
                        
                        # Include normality test results
                        if "normality_test" in dist_data:
                            f.write("**Normality Test**:\n\n")
                            norm_test = dist_data["normality_test"]
                            if "error" in norm_test:
                                f.write(f"- Error: {norm_test['error']}\n\n")
                            else:
                                is_normal = norm_test.get("is_normal", False)
                                p_value = norm_test.get("p_value", 0)
                                f.write(f"- Test: {norm_test.get('test', 'Unknown')}\n")
                                f.write(f"- P-value: {p_value}\n")
                                f.write(f"- Distribution is {'normal' if is_normal else 'not normal'}\n\n")
                        
                        # Include QQ plot if available - use just the filename
                        if "qq_plot_path" in dist_data and dist_data["qq_plot_path"]:
                            img_filename = Path(dist_data["qq_plot_path"]).name
                            f.write(f"**QQ Plot**:\n\n")
                            f.write(f"![QQ Plot - {col}]({img_filename})\n\n")
                        
                        # Include best distribution fit
                        if "best_fitting_distribution" in dist_data and dist_data["best_fitting_distribution"]:
                            f.write(f"**Best Fitting Distribution**: {dist_data['best_fitting_distribution']}\n\n")
                
                # Cluster Analysis
                if "cluster_analysis" in results["deep_analysis"] and results["deep_analysis"]["cluster_analysis"]:
                    f.write(f"### Cluster Analysis\n\n")
                    cluster_data = results["deep_analysis"]["cluster_analysis"]
                    
                    # Method and optimal clusters
                    f.write(f"**Method**: {cluster_data.get('method', 'K-Means')}\n\n")
                    f.write(f"**Optimal Number of Clusters**: {cluster_data.get('optimal_k', 'Not determined')}\n\n")
                    
                    # Include elbow plot if available
                    if "elbow_plot_path" in cluster_data:
                        file_path = Path(cluster_data["elbow_plot_path"])
                        img_filename = file_path.name
                        f.write(f"**Elbow Method Plot**:\n\n")
                        f.write(f"![Elbow Method]({img_filename})\n\n")
                    
                    # Include cluster plot if available
                    if "cluster_plot_path" in cluster_data:
                        file_path = Path(cluster_data["cluster_plot_path"])
                        img_filename = file_path.name
                        f.write(f"**Cluster Visualization**:\n\n")
                        f.write(f"![Cluster Visualization]({img_filename})\n\n")
                    
                    # Include cluster sizes
                    if "cluster_sizes" in cluster_data:
                        f.write("**Cluster Sizes**:\n\n")
                        for cluster, size in cluster_data["cluster_sizes"].items():
                            f.write(f"- {cluster}: {size} samples\n")
                        f.write("\n")
                    
                    # Include cluster centers
                    if "cluster_centers" in cluster_data:
                        f.write("**Cluster Centers**:\n\n")
                        f.write("```\n")
                        f.write(json.dumps(cluster_data["cluster_centers"], indent=2))
                        f.write("\n```\n\n")
                
                # Outlier Detection
                if "outlier_detection" in results["deep_analysis"] and results["deep_analysis"]["outlier_detection"]:
                    f.write(f"### Outlier Detection\n\n")
                    outlier_data = results["deep_analysis"]["outlier_detection"]
                    
                    for col, methods in outlier_data.items():
                        f.write(f"#### {col}\n\n")
                        
                        # IQR method
                        if "iqr_method" in methods:
                            f.write("**IQR Method**:\n\n")
                            iqr_data = methods["iqr_method"]
                            if "error" in iqr_data:
                                f.write(f"- Error: {iqr_data['error']}\n\n")
                            else:
                                f.write(f"- Lower bound: {iqr_data.get('lower_bound', 'N/A')}\n")
                                f.write(f"- Upper bound: {iqr_data.get('upper_bound', 'N/A')}\n")
                                f.write(f"- Outlier count: {iqr_data.get('outlier_count', 'N/A')}\n")
                                f.write(f"- Outlier percentage: {iqr_data.get('outlier_percentage', 'N/A')}%\n\n")
                        
                        # Z-score method
                        if "zscore_method" in methods:
                            f.write("**Z-Score Method (threshold=3)**:\n\n")
                            z_data = methods["zscore_method"]
                            if "error" in z_data:
                                f.write(f"- Error: {z_data['error']}\n\n")
                            else:
                                f.write(f"- Outlier count: {z_data.get('outlier_count', 'N/A')}\n")
                                f.write(f"- Outlier percentage: {z_data.get('outlier_percentage', 'N/A')}%\n\n")
                        
                        # Include boxplot if available
                        if "boxplot_path" in methods:
                            file_path = Path(methods["boxplot_path"])
                            img_filename = file_path.name
                            f.write(f"**Boxplot with Outliers**:\n\n")
                            f.write(f"![Boxplot - {col}]({img_filename})\n\n")
                
                # Time Series Analysis
                if "time_series_analysis" in results["deep_analysis"] and results["deep_analysis"]["time_series_analysis"]:
                    f.write(f"### Time Series Analysis\n\n")
                    ts_data = results["deep_analysis"]["time_series_analysis"]
                    
                    for dt_col, col_results in ts_data.items():
                        f.write(f"#### Time Series with {dt_col} as Index\n\n")
                        
                        for col, analysis in col_results.items():
                            f.write(f"##### {col}\n\n")
                            
                            # Include time series plot if available
                            if "time_series_plot" in analysis:
                                file_path = Path(analysis["time_series_plot"])
                                img_filename = file_path.name
                                f.write(f"**Time Series Plot**:\n\n")
                                f.write(f"![Time Series - {col}]({img_filename})\n\n")
                            
                            # Stationarity test results
                            if "stationarity_test" in analysis:
                                f.write("**Stationarity Test (Augmented Dickey-Fuller)**:\n\n")
                                stat_test = analysis["stationarity_test"]
                                if "error" in stat_test:
                                    f.write(f"- Error: {stat_test['error']}\n\n")
                                else:
                                    is_stationary = stat_test.get("is_stationary", False)
                                    p_value = stat_test.get("p_value", 0)
                                    f.write(f"- P-value: {p_value}\n")
                                    f.write(f"- Series is {'stationary' if is_stationary else 'non-stationary'}\n\n")
                            
                            # Seasonal decomposition plot
                            if "seasonal_decomposition" in analysis and "plot_path" in analysis["seasonal_decomposition"]:
                                file_path = Path(analysis["seasonal_decomposition"]["plot_path"])
                                img_filename = file_path.name
                                f.write(f"**Seasonal Decomposition**:\n\n")
                                f.write(f"![Decomposition - {col}]({img_filename})\n\n")
                
                # Feature Importance
                if "feature_importance" in results["deep_analysis"] and results["deep_analysis"]["feature_importance"]:
                    f.write(f"### Feature Importance Analysis\n\n")
                    fi_data = results["deep_analysis"]["feature_importance"]
                    
                    for target, analysis in fi_data.items():
                        f.write(f"#### Predicting {target}\n\n")
                        
                        if "error" in analysis:
                            f.write(f"- Error: {analysis['error']}\n\n")
                            continue
                        
                        # Include feature importance plot if available
                        if "plot_path" in analysis:
                            file_path = Path(analysis["plot_path"])
                            img_filename = file_path.name
                            f.write(f"**Feature Importance Plot**:\n\n")
                            f.write(f"![Feature Importance - {target}]({img_filename})\n\n")
                        
                        # Cross-validation results
                        if "cv_rmse" in analysis:
                            f.write(f"**Cross-Validation Performance (RMSE)**: {analysis['cv_rmse']:.4f} ± {analysis.get('cv_rmse_std', 0):.4f}\n\n")
                        
                        # Top important features
                        if "feature_importance" in analysis:
                            f.write("**Top Features**:\n\n")
                            f.write("| Feature | Importance |\n")
                            f.write("|---------|------------|\n")
                            for item in analysis["feature_importance"][:5]:  # Show top 5
                                f.write(f"| {item['feature']} | {item['importance']:.4f} |\n")
                            f.write("\n")
                
                # Correlation Network
                if "correlation_network" in results["deep_analysis"] and "plot_path" in results["deep_analysis"]["correlation_network"]:
                    f.write(f"### Correlation Network\n\n")
                    network_data = results["deep_analysis"]["correlation_network"]
                    
                    if "error" in network_data:
                        f.write(f"- Error: {network_data['error']}\n\n")
                    else:
                        # Include correlation threshold
                        if "threshold" in network_data:
                            f.write(f"**Correlation Threshold**: {network_data['threshold']}\n\n")
                        
                        # Include network visualization
                        if "plot_path" in network_data:
                            file_path = Path(network_data["plot_path"])
                            img_filename = file_path.name
                            f.write(f"**Correlation Network Visualization**:\n\n")
                            f.write(f"![Correlation Network]({img_filename})\n\n")
                        
                        # Include top correlations
                        if "correlations" in network_data and network_data["correlations"]:
                            f.write("**Top Correlations**:\n\n")
                            f.write("| Feature 1 | Feature 2 | Correlation |\n")
                            f.write("|-----------|-----------|-------------|\n")
                            for corr in network_data["correlations"][:10]:  # Show top 10
                                f.write(f"| {corr['source']} | {corr['target']} | {corr['correlation']:.4f} |\n")
                            f.write("\n")
                
                # Hypothesis Tests
                if "hypothesis_tests" in results["deep_analysis"] and results["deep_analysis"]["hypothesis_tests"]:
                    f.write(f"### Hypothesis Tests\n\n")
                    hypothesis_data = results["deep_analysis"]["hypothesis_tests"]
                    
                    # Correlation tests
                    if "correlation_tests" in hypothesis_data and hypothesis_data["correlation_tests"]:
                        f.write(f"#### Correlation Tests\n\n")
                        f.write("| Variables | Correlation | P-value | Significant |\n")
                        f.write("|-----------|-------------|---------|-------------|\n")
                        
                        for test_name, test_data in hypothesis_data["correlation_tests"].items():
                            variables = " vs ".join(test_data.get("columns", [test_name]))
                            correlation = test_data.get("correlation", "N/A")
                            p_value = test_data.get("p_value", "N/A")
                            significant = "Yes" if test_data.get("significant", False) else "No"
                            
                            f.write(f"| {variables} | {correlation:.4f} | {p_value:.4f} | {significant} |\n")
                        f.write("\n")
                    
                    # Group difference tests
                    if "group_difference_tests" in hypothesis_data and hypothesis_data["group_difference_tests"]:
                        f.write(f"#### Group Difference Tests\n\n")
                        
                        for test_name, test_data in hypothesis_data["group_difference_tests"].items():
                            test_type = test_data.get("test", "Unknown")
                            cat_col = test_data.get("categorical_column", "Unknown")
                            num_col = test_data.get("numeric_column", "Unknown")
                            
                            f.write(f"##### {test_type.upper()}: {num_col} by {cat_col}\n\n")
                            
                            if "error" in test_data:
                                f.write(f"- Error: {test_data['error']}\n\n")
                                continue
                            
                            statistic = test_data.get("f_statistic" if test_type == "anova" else "t_statistic", "N/A")
                            if isinstance(statistic, (int, float)):
                                f.write(f"- Statistic: {statistic:.4f}\n")
                            else:
                                f.write(f"- Statistic: {statistic}\n")
                            
                            p_value = test_data.get("p_value", "N/A")
                            if isinstance(p_value, (int, float)):
                                f.write(f"- P-value: {p_value:.4f}\n")
                            else:
                                f.write(f"- P-value: {p_value}\n")
                            
                            significant = "Yes" if test_data.get("significant", False) else "No"
                            interpretation = test_data.get("interpretation", "")
                            
                            f.write(f"- Significant: {significant}\n")
                            f.write(f"- Interpretation: {interpretation}\n\n")
            
            # Insights
            f.write(f"## Key Insights\n\n")
            for i, insight in enumerate(results['insights']):
                f.write(f"{i+1}. {insight}\n")
            
            # Add footer with generated timestamp
            f.write(f"\n\n---\n\n")
            f.write(f"*This report was automatically generated with Smart EDA using Azure OpenAI GPT-4o-mini at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        return report_path
    
    def generate_html(self, df, metadata, eda_plan, results):
        """
        Generate an HTML report of the EDA
        
        Args:
            df: Pandas DataFrame
            metadata: Dictionary with dataset metadata
            eda_plan: Dictionary with the EDA plan
            results: Dictionary with EDA results
            
        Returns:
            str: Path to the generated report
        """
        report_path = self.output_dir / "eda_report.html"
        
        # Convert markdown to HTML
        markdown_path = self.generate_markdown(df, metadata, eda_plan, results)
        
        try:
            import markdown
            with open(markdown_path, 'r') as f:
                md_content = f.read()
                
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>EDA Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    h1 {{ color: #2c3e50; }}
                    h2, h3 {{ color: #3498db; }}
                    h4, h5 {{ color: #2980b9; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; overflow-x: auto; display: block; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; }}
                    code {{ background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    .toc {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .toc ul {{ list-style-type: none; padding-left: 15px; }}
                    .toc li {{ margin-bottom: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                {markdown.markdown(md_content, extensions=['tables', 'fenced_code'])}
                </div>
                
                <script>
                // Add table of contents
                window.addEventListener('load', function() {{
                    const container = document.querySelector('.container');
                    const h1 = document.querySelector('h1');
                    const toc = document.createElement('div');
                    toc.className = 'toc';
                    toc.innerHTML = '<h3>Table of Contents</h3><ul></ul>';
                    
                    const headings = document.querySelectorAll('h2');
                    headings.forEach(function(heading) {{
                        const li = document.createElement('li');
                        const a = document.createElement('a');
                        a.textContent = heading.textContent;
                        a.href = '#' + heading.textContent.toLowerCase().replace(/\\s+/g, '-');
                        heading.id = heading.textContent.toLowerCase().replace(/\\s+/g, '-');
                        li.appendChild(a);
                        toc.querySelector('ul').appendChild(li);
                    }});
                    
                    container.insertBefore(toc, h1.nextSibling);
                }});
                </script>
            </body>
            </html>
            """
            
            with open(report_path, 'w') as f:
                f.write(html)
                
            return report_path
        except ImportError:
            print("Python-markdown package not found. Falling back to markdown report.")
            return markdown_path

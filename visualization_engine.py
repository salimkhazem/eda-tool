import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import json
from datetime import datetime
import io
import base64
from statsmodels.graphics.mosaicplot import mosaic

class VisualizationEngine:
    """Advanced visualization engine that creates static and interactive plots."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.interactive_dir = self.output_dir / "interactive"
        self.interactive_dir.mkdir(exist_ok=True)
        
        # Set default style with enhanced aesthetics - updated color schemes
        sns.set_theme(style="whitegrid", context="notebook")
        # Use a more professional and visually appealing color palette
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#4C72B0', '#55A868', '#C44E52', 
                                                            '#8172B3', '#CCB974', '#64B5CD', 
                                                            '#4C72B0', '#55A868', '#C44E52'])
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Store visualization metadata
        self.visualization_results = []
    
    def create_visualization(self, df, viz_config):
        """
        Create a visualization based on configuration
        
        Args:
            df: Pandas DataFrame
            viz_config: Dictionary with visualization configuration
            
        Returns:
            dict: Visualization metadata
        """
        plot_type = viz_config.get("type", "").lower()
        title = viz_config.get("title", f"Plot - {plot_type}")
        columns = viz_config.get("columns", [])
        description = viz_config.get("description", "")
        interactive = viz_config.get("interactive", False)
        
        # Skip if required columns don't exist
        if not all(col in df.columns for col in columns):
            return None
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_id = f"{len(self.visualization_results) + 1:03d}_{timestamp}"
        
        try:
            # Choose between static matplotlib/seaborn or interactive plotly
            if interactive:
                result = self._create_interactive_plot(df, viz_config, file_id)
            else:
                result = self._create_static_plot(df, viz_config, file_id)
                
            # Add metadata if plot was created successfully
            if result:
                self.visualization_results.append(result)
                return result
                
        except Exception as e:
            print(f"Error creating {plot_type} plot for {columns}: {e}")
            return None
    
    def _create_static_plot(self, df, viz_config, file_id):
        """Create a static matplotlib/seaborn plot"""
        plot_type = viz_config.get("type", "").lower()
        title = viz_config.get("title", f"Plot - {plot_type}")
        columns = viz_config.get("columns", [])
        description = viz_config.get("description", "")
        palette = viz_config.get("palette", "viridis")
        
        # Create new figure
        plt.figure(figsize=(12, 8))
        
        file_path = self.figures_dir / f"{file_id}_{plot_type}.png"
        plot_created = False
        
        # Univariate plots
        if plot_type == "histogram":
            if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                sns.histplot(df[columns[0]].dropna(), kde=True, bins=30, 
                            color='#4C72B0', edgecolor='white', alpha=0.7)
                plt.xlabel(columns[0])
                plt.ylabel("Frequency")
                plot_created = True
        
        elif plot_type == "density" or plot_type == "kde":
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                for i, col in enumerate(numeric_cols):
                    # Use consistent colors from the color cycle
                    sns.kdeplot(df[col].dropna(), label=col, fill=True, alpha=0.3)
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.legend()
                plot_created = True
        
        elif plot_type == "boxplot" or plot_type == "box":
            if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                sns.boxplot(y=df[columns[0]])
                plt.ylabel(columns[0])
                plot_created = True
            elif len(columns) == 2:
                if pd.api.types.is_numeric_dtype(df[columns[0]]) and not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    sns.boxplot(x=df[columns[1]], y=df[columns[0]])
                    plt.xlabel(columns[1])
                    plt.ylabel(columns[0])
                    plt.xticks(rotation=45, ha='right')
                    plot_created = True
                elif pd.api.types.is_numeric_dtype(df[columns[1]]) and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                    sns.boxplot(x=df[columns[0]], y=df[columns[1]])
                    plt.xlabel(columns[0])
                    plt.ylabel(columns[1])
                    plt.xticks(rotation=45, ha='right')
                    plot_created = True
        
        elif plot_type == "violin":
            if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                sns.violinplot(y=df[columns[0]].dropna())
                plt.ylabel(columns[0])
                plot_created = True
            elif len(columns) == 2:
                if pd.api.types.is_numeric_dtype(df[columns[0]]) and not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    sns.violinplot(x=df[columns[1]], y=df[columns[0]])
                    plt.xlabel(columns[1])
                    plt.ylabel(columns[0])
                    plt.xticks(rotation=45, ha='right')
                    plot_created = True
                elif pd.api.types.is_numeric_dtype(df[columns[1]]) and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                    sns.violinplot(x=df[columns[0]], y=df[columns[1]])
                    plt.xlabel(columns[0])
                    plt.ylabel(columns[1])
                    plt.xticks(rotation=45, ha='right')
                    plot_created = True
        
        elif plot_type == "bar":
            if len(columns) == 1:
                value_counts = df[columns[0]].value_counts().sort_values(ascending=False).head(10)
                value_counts.plot(kind='bar')
                plt.xlabel(columns[0])
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                plot_created = True
            elif len(columns) == 2:
                pd.crosstab(df[columns[0]], df[columns[1]]).plot(kind='bar', stacked=False)
                plt.xlabel(columns[0])
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                plt.legend(title=columns[1], bbox_to_anchor=(1.05, 1), loc='upper left')
                plot_created = True
        
        elif plot_type == "countplot":
            if len(columns) >= 1 and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                # Limit to top N categories if too many
                value_counts = df[columns[0]].value_counts()
                if len(value_counts) > 15:
                    top_categories = value_counts.head(15).index
                    data = df[df[columns[0]].isin(top_categories)].copy()
                    data[columns[0]] = data[columns[0]].astype(str)
                else:
                    data = df.copy()
                
                if len(columns) >= 2:  # Use second column for hue
                    sns.countplot(x=data[columns[0]], hue=data[columns[1]])
                    plt.legend(title=columns[1], bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    sns.countplot(x=data[columns[0]])
                    
                plt.xlabel(columns[0])
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Count')
                plot_created = True
        
        elif plot_type == "pie":
            if len(columns) == 1 and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                counts = df[columns[0]].value_counts()
                # Limit to top categories if too many
                if len(counts) > 8:
                    other_count = counts[7:].sum()
                    counts = counts[:7]
                    counts['Other'] = other_count
                plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                plot_created = True
        
        # Bivariate plots
        elif plot_type == "scatter":
            if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                if len(columns) >= 3:
                    sns.scatterplot(x=df[columns[0]], y=df[columns[1]], hue=df[columns[2]])
                    plt.legend(title=columns[2], bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    sns.scatterplot(x=df[columns[0]], y=df[columns[1]])
                plt.xlabel(columns[0])
                plt.ylabel(columns[1])
                plot_created = True
        
        elif plot_type == "line":
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
                    plot_created = True
        
        elif plot_type == "regplot":
            if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                sns.regplot(x=df[columns[0]], y=df[columns[1]], scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
                plt.xlabel(columns[0])
                plt.ylabel(columns[1])
                plot_created = True
        
        elif plot_type == "hexbin":
            if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                plt.hexbin(df[columns[0]], df[columns[1]], gridsize=30, cmap='viridis')
                plt.colorbar(label='Count')
                plt.xlabel(columns[0])
                plt.ylabel(columns[1])
                plot_created = True
        
        # Multivariate plots
        elif plot_type == "heatmap" or plot_type == "correlation":
            corr_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(corr_cols) > 1:
                corr_matrix = df[corr_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                # Use a diverging colormap that's easier to interpret
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f', 
                            mask=mask, vmin=-1, vmax=1, square=True,
                            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})
                plot_created = True
        
        elif plot_type == "pairplot":
            plot_cols = columns[:5] if len(columns) > 5 else columns
            plt.close()  # Close the current figure as pairplot creates its own
            g = sns.pairplot(df[plot_cols])
            g.fig.suptitle(title, y=1.02)
            g.fig.tight_layout()
            g.fig.savefig(file_path)
            plt.close(g.fig)
            plot_created = True
            return {
                "title": title,
                "description": description,
                "file_path": str(file_path),
                "type": plot_type,
                "columns": columns,
                "interactive": False
            }
        
        elif plot_type == "jointplot":
            if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                plt.close()  # Close the current figure as jointplot creates its own
                kind = "scatter"
                if len(df) > 1000:
                    kind = "hex"  # Use hex plot for large datasets
                g = sns.jointplot(x=df[columns[0]], y=df[columns[1]], kind=kind)
                g.fig.suptitle(title, y=1.02)
                g.fig.tight_layout()
                g.fig.savefig(file_path)
                plt.close(g.fig)
                plot_created = True
                return {
                    "title": title,
                    "description": description,
                    "file_path": str(file_path),
                    "type": plot_type,
                    "columns": columns,
                    "interactive": False
                }
        
        elif plot_type == "ridgeplot":
            if len(columns) == 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and not pd.api.types.is_numeric_dtype(df[columns[1]]):
                # Determine the top categories if there are too many
                categories = df[columns[1]].value_counts().head(8).index
                
                # Create the ridgeplot
                for i, category in enumerate(categories):
                    sns.kdeplot(
                        df[df[columns[1]] == category][columns[0]].dropna(),
                        label=category,
                        fill=True,
                        alpha=0.5
                    )
                
                plt.legend(title=columns[1])
                plt.xlabel(columns[0])
                plt.ylabel('Density')
                plot_created = True
        
        elif plot_type == "barh":
            if len(columns) == 1 and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                # Horizontal bar plot for categories
                value_counts = df[columns[0]].value_counts().sort_values()
                if len(value_counts) > 15:  # Limit to top 15 for readability
                    value_counts = value_counts.tail(15)
                value_counts.plot(kind='barh')
                plt.xlabel('Count')
                plt.ylabel(columns[0])
                plt.tight_layout()
                plot_created = True
        
        # Additional seaborn plots
        elif plot_type == "pairplot" or plot_type == "pair":
            plot_cols = columns[:6] if len(columns) > 6 else columns  # Limit to 6 columns for readability
            plt.close()  # Close the current figure as pairplot creates its own

            # Check if there's a categorical column to use as hue
            categorical_cols = [c for c in plot_cols if not pd.api.types.is_numeric_dtype(df[c])]
            if categorical_cols and len(categorical_cols) == 1 and len(plot_cols) > 2:
                hue_col = categorical_cols[0]
                numeric_cols = [c for c in plot_cols if c != hue_col]
                
                # Limit hue categories if there are too many
                if df[hue_col].nunique() > 5:
                    top_categories = df[hue_col].value_counts().head(5).index
                    plot_df = df[df[hue_col].isin(top_categories)].copy()
                else:
                    plot_df = df.copy()
                
                g = sns.pairplot(plot_df, vars=numeric_cols, hue=hue_col, 
                                corner=True, diag_kind="kde", plot_kws={"alpha": 0.6})
            else:
                # Basic pairplot with only numeric columns
                numeric_cols = [c for c in plot_cols if pd.api.types.is_numeric_dtype(df[c])]
                if len(numeric_cols) < 2:
                    plt.close()
                    return None
                    
                g = sns.pairplot(df[numeric_cols], corner=True, diag_kind="kde", plot_kws={"alpha": 0.6})
            
            g.fig.suptitle(title, y=1.02)
            g.fig.tight_layout()
            g.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            
            return {
                "title": title,
                "description": description,
                "file_path": str(file_path),
                "type": plot_type,
                "columns": columns,
                "interactive": False
            }
        
        elif plot_type == "heatmap_annot" or plot_type == "annotated_heatmap":
            # Create a more readable correlation heatmap with annotations
            corr_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(corr_cols) > 1:
                corr_matrix = df[corr_cols].corr()
                
                # Create a mask for the upper triangle to avoid redundancy
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Draw the heatmap with the mask and correct aspect ratio
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                            cmap="coolwarm", center=0, square=True, linewidths=.5,
                            cbar_kws={"shrink": .8, "label": "Correlation Coefficient"})
                
                plt.title(title)
                plt.tight_layout()
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_created = True
        
        elif plot_type == "catplot":
            # Specialized plot for categorical data
            if len(columns) >= 2:
                cat_col = None
                num_col = None
                
                # Identify categorical and numeric columns
                for col in columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        cat_col = col
                    else:
                        num_col = col
                        
                if cat_col and num_col:
                    # Close current figure as catplot creates its own
                    plt.close()
                    
                    # Use multiple different visualizations in a FacetGrid
                    g = sns.catplot(
                        data=df, x=cat_col, y=num_col,
                        kind="box", height=6, aspect=1.5,
                        palette=palette
                    )
                    g.fig.suptitle(title, y=1.02)
                    g.set_xticklabels(rotation=45, ha="right")
                    g.tight_layout()
                    g.savefig(file_path, dpi=300, bbox_inches='tight')
                    plt.close(g.fig)
                    
                    return {
                        "title": title,
                        "description": description,
                        "file_path": str(file_path),
                        "type": plot_type,
                        "columns": columns,
                        "interactive": False
                    }
        
        elif plot_type == "lmplot":
            # Regression plot with confidence intervals
            if len(columns) >= 2 and all(pd.api.types.is_numeric_dtype(df[col]) for col in columns[:2]):
                x_col, y_col = columns[:2]
                
                # Get hue column if available
                hue_col = None
                if len(columns) > 2 and not pd.api.types.is_numeric_dtype(df[columns[2]]):
                    hue_col = columns[2]
                
                plt.close()  # Close current figure
                
                if hue_col:
                    # Limit to top categories if there are too many
                    if df[hue_col].nunique() > 5:
                        top_cats = df[hue_col].value_counts().head(5).index
                        plot_df = df[df[hue_col].isin(top_cats)]
                    else:
                        plot_df = df
                        
                    g = sns.lmplot(
                        data=plot_df, x=x_col, y=y_col, hue=hue_col,
                        height=7, scatter_kws={"alpha": 0.6}, 
                        line_kws={"linewidth": 2}
                    )
                else:
                    g = sns.lmplot(
                        data=df, x=x_col, y=y_col,
                        height=7, scatter_kws={"alpha": 0.6}, 
                        line_kws={"linewidth": 2}
                    )
                
                g.fig.suptitle(title, y=1.02)
                g.tight_layout()
                g.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close(g.fig)
                
                return {
                    "title": title,
                    "description": description,
                    "file_path": str(file_path),
                    "type": plot_type,
                    "columns": columns,
                    "interactive": False
                }
        
        elif plot_type == "facet_grid":
            # Create a facet grid with multiple panels
            if len(columns) >= 3:
                # Need at least one numeric and two categorical columns
                numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
                cat_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
                
                if numeric_cols and len(cat_cols) >= 2:
                    num_col = numeric_cols[0]
                    cat_col1 = cat_cols[0]
                    cat_col2 = cat_cols[1]
                    
                    # Close current figure
                    plt.close()
                    
                    # Limit categories if there are too many
                    if df[cat_col1].nunique() > 5 or df[cat_col2].nunique() > 5:
                        top_cats1 = df[cat_col1].value_counts().head(5).index
                        top_cats2 = df[cat_col2].value_counts().head(5).index
                        plot_df = df[df[cat_col1].isin(top_cats1) & df[cat_col2].isin(top_cats2)]
                    else:
                        plot_df = df
                    
                    g = sns.FacetGrid(plot_df, row=cat_col1, col=cat_col2, margin_titles=True, height=3.5, aspect=1.2)
                    g.map(sns.histplot, num_col, kde=True)
                    g.fig.suptitle(title, y=1.02)
                    g.tight_layout()
                    g.savefig(file_path, dpi=300, bbox_inches='tight')
                    plt.close(g.fig)
                    
                    return {
                        "title": title,
                        "description": description,
                        "file_path": str(file_path),
                        "type": plot_type,
                        "columns": columns,
                        "interactive": False
                    }
        
        elif plot_type == "displot":
            # Enhanced distribution plot
            if len(columns) >= 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                num_col = columns[0]
                
                # Get hue column if available
                hue_col = None
                if len(columns) > 1 and not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    hue_col = columns[1]
                
                plt.close()  # Close current figure
                
                if hue_col:
                    # Limit categories if there are too many
                    if df[hue_col].nunique() > 5:
                        top_cats = df[hue_col].value_counts().head(5).index
                        plot_df = df[df[hue_col].isin(top_cats)]
                    else:
                        plot_df = df
                        
                    g = sns.displot(
                        data=plot_df, x=num_col, hue=hue_col,
                        kind="kde", height=6, aspect=1.5, 
                        fill=True, common_norm=False, alpha=0.5
                    )
                else:
                    g = sns.displot(
                        data=df, x=num_col,
                        kind="kde", height=6, aspect=1.5, fill=True
                    )
                
                g.fig.suptitle(title, y=1.02)
                g.tight_layout()
                g.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close(g.fig)
                
                return {
                    "title": title,
                    "description": description,
                    "file_path": str(file_path),
                    "type": plot_type,
                    "columns": columns,
                    "interactive": False
                }
        
        elif plot_type == "violinplot":
            # Enhanced violin plot
            if len(columns) >= 2:
                # One numeric and one categorical column
                num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
                cat_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
                
                if num_cols and cat_cols:
                    num_col = num_cols[0]
                    cat_col = cat_cols[0]
                    
                    # Get split column if available (for split violins)
                    split_col = None
                    if len(cat_cols) > 1:
                        split_col = cat_cols[1]
                    
                    # Limit categories if there are too many
                    if df[cat_col].nunique() > 8:
                        top_cats = df[cat_col].value_counts().head(8).index
                        plot_df = df[df[cat_col].isin(top_cats)]
                    else:
                        plot_df = df
                    
                    plt.figure(figsize=(12, 8))
                    if split_col and df[split_col].nunique() <= 2:
                        sns.violinplot(x=cat_col, y=num_col, hue=split_col, 
                                      split=True, data=plot_df, palette=palette)
                        plt.legend(title=split_col)
                    else:
                        sns.violinplot(x=cat_col, y=num_col, data=plot_df, palette=palette)
                    
                    # Add stripplot for individual data points
                    sns.stripplot(x=cat_col, y=num_col, data=plot_df, size=4, 
                                 color="black", alpha=0.3)
                    
                    plt.title(title)
                    plt.xlabel(cat_col)
                    plt.ylabel(num_col)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    return {
                        "title": title,
                        "description": description,
                        "file_path": str(file_path),
                        "type": plot_type,
                        "columns": columns,
                        "interactive": False
                    }
        
        # Finalize the plot with better styling
        if plot_created:
            plt.title(title, fontweight='bold')
            # Add subtle grid lines for better readability
            plt.grid(True, alpha=0.2, linestyle='--')
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "title": title,
                "description": description,
                "file_path": str(file_path),
                "type": plot_type,
                "columns": columns,
                "interactive": False
            }
        
    def _create_interactive_plot(self, df, viz_config, file_id):
        """Create an interactive plotly plot"""
        plot_type = viz_config.get("type", "").lower()
        title = viz_config.get("title", f"Plot - {plot_type}")
        columns = viz_config.get("columns", [])
        description = viz_config.get("description", "")
        
        file_path = self.interactive_dir / f"{file_id}_{plot_type}.html"
        
        # Use improved color schemes for interactive plots
        custom_colorscale = px.colors.sequential.Plasma
        custom_diverging = px.colors.diverging.RdBu_r
        custom_qualitative = px.colors.qualitative.G10
        
        # Univariate plots
        if plot_type == "histogram":
            if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                fig = px.histogram(df, x=columns[0], nbins=30, title=title,
                                color_discrete_sequence=[custom_qualitative[0]],
                                opacity=0.8)
                fig.update_layout(template="plotly_white")
        
        elif plot_type == "density" or plot_type == "kde":
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                fig = go.Figure()
                for col in numeric_cols:
                    fig.add_trace(go.Histogram(x=df[col].dropna(), histnorm='density', name=col, opacity=0.6))
                fig.update_layout(barmode='overlay', title=title)
        
        elif plot_type == "boxplot" or plot_type == "box":
            if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                fig = px.box(df, y=columns[0], title=title)
            elif len(columns) == 2:
                if pd.api.types.is_numeric_dtype(df[columns[0]]) and not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    fig = px.box(df, x=columns[1], y=columns[0], title=title)
                elif pd.api.types.is_numeric_dtype(df[columns[1]]) and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                    fig = px.box(df, x=columns[0], y=columns[1], title=title)
        
        elif plot_type == "violin":
            if len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                fig = px.violin(df, y=columns[0], box=True, points="all", title=title)
            elif len(columns) == 2:
                if pd.api.types.is_numeric_dtype(df[columns[0]]) and not pd.api.types.is_numeric_dtype(df[columns[1]]):
                    fig = px.violin(df, x=columns[1], y=columns[0], box=True, points="all", title=title)
                elif pd.api.types.is_numeric_dtype(df[columns[1]]) and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                    fig = px.violin(df, x=columns[0], y=columns[1], box=True, points="all", title=title)
        
        elif plot_type == "bar":
            if len(columns) == 1:
                value_counts = df[columns[0]].value_counts().sort_values(ascending=False).head(10)
                fig = px.bar(value_counts, x=value_counts.index, y=value_counts.values, title=title)
            elif len(columns) == 2:
                crosstab = pd.crosstab(df[columns[0]], df[columns[1]])
                fig = px.bar(crosstab, x=crosstab.index, y=crosstab.columns, title=title)
        
        elif plot_type == "pie":
            if len(columns) == 1 and not pd.api.types.is_numeric_dtype(df[columns[0]]):
                counts = df[columns[0]].value_counts()
                if len(counts) > 8:
                    other_count = counts[7:].sum()
                    counts = counts[:7]
                    counts['Other'] = other_count
                fig = px.pie(counts, values=counts.values, names=counts.index, title=title)
        
        # Bivariate plots
        elif plot_type == "scatter":
            if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                if len(columns) >= 3:
                    fig = px.scatter(df, x=columns[0], y=columns[1], color=columns[2], 
                                    title=title, color_discrete_sequence=custom_qualitative,
                                    opacity=0.8, template="plotly_white")
                else:
                    fig = px.scatter(df, x=columns[0], y=columns[1], title=title,
                                    color_discrete_sequence=[custom_qualitative[0]],
                                    opacity=0.8, template="plotly_white")
        
        elif plot_type == "line":
            if len(columns) >= 2:
                if pd.api.types.is_datetime64_any_dtype(df[columns[0]]) or isinstance(df[columns[0]].iloc[0], (datetime, str)):
                    if isinstance(df[columns[0]].iloc[0], str):
                        try:
                            df[columns[0]] = pd.to_datetime(df[columns[0]])
                        except:
                            pass
                    fig = go.Figure()
                    for col in columns[1:]:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig.add_trace(go.Scatter(x=df[columns[0]], y=df[col], mode='lines', name=col))
                    fig.update_layout(title=title)
        
        elif plot_type == "regplot":
            if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                fig = px.scatter(df, x=columns[0], y=columns[1], trendline="ols", title=title)
        
        elif plot_type == "hexbin":
            if len(columns) >= 2 and pd.api.types.is_numeric_dtype(df[columns[0]]) and pd.api.types.is_numeric_dtype(df[columns[1]]):
                fig = px.density_heatmap(df, x=columns[0], y=columns[1], nbinsx=30, nbinsy=30, title=title)
        
        # Multivariate plots
        elif plot_type == "heatmap" or plot_type == "correlation":
            corr_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(corr_cols) > 1:
                corr_matrix = df[corr_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title=title,
                            color_continuous_scale=custom_diverging, zmin=-1, zmax=1)
                fig.update_layout(template="plotly_white")
        
        elif plot_type == "scattermatrix":
            plot_cols = columns[:5] if len(columns) > 5 else columns
            fig = px.scatter_matrix(df[plot_cols], title=title)
        
        elif plot_type == "bubble":
            if len(columns) >= 3 and all(pd.api.types.is_numeric_dtype(df[col]) for col in columns[:3]):
                x_col, y_col, size_col = columns[:3]
                
                # Get color column if available
                color_col = None
                if len(columns) > 3:
                    color_col = columns[3]
                
                fig = px.scatter(df, x=x_col, y=y_col, size=size_col, 
                                color=color_col if color_col else None,
                                hover_name=df.index if df.index.name else None,
                                size_max=40, opacity=0.7,
                                title=title)
        
        elif plot_type == "parallel_coordinates":
            # Select only numeric columns and a single categorical for color
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
            
            if numeric_cols and len(numeric_cols) >= 3:
                # Use up to 8 numeric dimensions for readability
                plot_cols = numeric_cols[:8]
                
                # Add categorical column for color if available
                color_col = None
                if cat_cols:
                    color_col = cat_cols[0]
                    # Limit to top categories if too many
                    if df[color_col].nunique() > 8:
                        top_cats = df[color_col].value_counts().head(8).index
                        plot_df = df[df[color_col].isin(top_cats)].copy()
                    else:
                        plot_df = df.copy()
                else:
                    plot_df = df.copy()
                
                fig = px.parallel_coordinates(
                    plot_df, 
                    dimensions=plot_cols,
                    color=color_col if color_col else None,
                    title=title,
                    color_continuous_scale=px.colors.diverging.Tealrose if color_col and pd.api.types.is_numeric_dtype(df[color_col]) else None
                )
                
                fig.update_layout(coloraxis_colorbar=dict(title=color_col if color_col else ""))
        
        elif plot_type == "violin":
            if len(columns) >= 2:
                # Find categorical and numeric columns
                cat_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
                num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
                
                if cat_cols and num_cols:
                    # Limit categories if needed
                    if df[cat_cols[0]].nunique() > 10:
                        top_cats = df[cat_cols[0]].value_counts().head(10).index
                        plot_df = df[df[cat_cols[0]].isin(top_cats)]
                    else:
                        plot_df = df
                    
                    # Create violin plot with box inside
                    fig = px.violin(
                        plot_df, 
                        x=cat_cols[0], 
                        y=num_cols[0],
                        color=cat_cols[1] if len(cat_cols) > 1 else None,
                        box=True, 
                        points="all",
                        title=title
                    )
        
        elif plot_type == "treemap" or plot_type == "sunburst":
            # These plots work best with hierarchical categorical data
            cat_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
            num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            
            if len(cat_cols) >= 1:
                # Use categorical columns as path and numeric column as value if available
                path = cat_cols[:3]  # Limit to 3 levels for readability
                values = num_cols[0] if num_cols else None
                
                # Sample if dataset is very large (for better performance)
                plot_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
                
                if plot_type == "treemap":
                    fig = px.treemap(
                        plot_df,
                        path=path,
                        values=values,
                        color=path[-1],
                        title=title
                    )
                else:  # sunburst
                    fig = px.sunburst(
                        plot_df,
                        path=path,
                        values=values,
                        color=path[-1],
                        title=title
                    )
        
        # Save the interactive plot
        if fig:
            fig.write_html(str(file_path))
            return {
                "title": title,
                "description": description,
                "file_path": str(file_path),
                "type": plot_type,
                "columns": columns,
                "interactive": True
            }
        
        return None
    
    def get_visualization_dashboard(self):
        """Generate an HTML dashboard with all visualizations"""
        dashboard_path = self.output_dir / "visualization_dashboard.html"
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Visualization Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                .container { width: 90%; margin: 0 auto; padding: 20px; }
                .tab { cursor: pointer; padding: 10px 20px; display: inline-block; background: #f1f1f1; margin-right: 5px; }
                .tab.active { background: #007bff; color: white; }
                .tab-content { display: none; padding: 20px 0; }
                .tab-content.active { display: block; }
                .plot-container { margin-bottom: 40px; }
                .plot-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
                .plot-description { font-size: 14px; margin-bottom: 10px; }
                .plot-image { text-align: center; }
                .plot-image img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Visualization Dashboard</h1>
                <div class="tabs">
                    <div class="tab active" onclick="openTab(event, 'univariate')">Univariate</div>
                    <div class="tab" onclick="openTab(event, 'bivariate')">Bivariate</div>
                    <div class="tab" onclick="openTab(event, 'multivariate')">Multivariate</div>
                </div>
        """
        
        # Add univariate tab content
        html_content += '<div id="univariate" class="tab-content active">'
        
        univariate_types = ["histogram", "density", "kde", "boxplot", "box", "violin", "bar", "countplot", "pie"]
        bivariate_types = ["scatter", "line", "regplot", "hexbin"]
        multivariate_types = ["heatmap", "correlation", "pairplot", "jointplot", "ridgeplot", "barh", "pair", "heatmap_annot", "annotated_heatmap", "catplot", "lmplot", "facet_grid", "displot", "scattermatrix", "bubble", "parallel_coordinates", "treemap", "sunburst"]
        
        for viz in self.visualization_results:
            if viz['type'].lower() in univariate_types:
                html_content += self._create_viz_html(viz)
        
        html_content += '</div>'
        
        # Add bivariate tab content
        html_content += '<div id="bivariate" class="tab-content">'
        
        for viz in self.visualization_results:
            if viz['type'].lower() in bivariate_types:
                html_content += self._create_viz_html(viz)
        
        html_content += '</div>'
        
        # Add multivariate tab content
        html_content += '<div id="multivariate" class="tab-content">'
        
        for viz in self.visualization_results:
            if viz['type'].lower() in multivariate_types:
                html_content += self._create_viz_html(viz)
        
        html_content += '</div>'
        
        # Add JavaScript for tab functionality
        html_content += """
                <script>
                function openTab(evt, tabName) {
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                        tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                    }
                    
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }
                    
                    document.getElementById(tabName).style.display = "block";
                    document.getElementById(tabName).className += " active";
                    evt.currentTarget.className += " active";
                }
                </script>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        return str(dashboard_path)
    
    def _create_viz_html(self, viz):
        """Helper to create HTML for a single visualization"""
        html = f"""
        <div class="plot-container">
            <div class="plot-title">{viz['title']}</div>
            <div class="plot-description">{viz['description']}</div>
            <div class="plot-image">
        """
        
        if viz['interactive'] and Path(viz['file_path']).exists():
            # For interactive plots, embed an iframe
            html += f'<iframe src="{Path(viz["file_path"]).name}" width="100%" height="500px" frameborder="0"></iframe>'
        else:
            # For static plots, use img tag
            img_path = viz.get('static_path', viz['file_path'])
            if Path(img_path).exists():
                rel_path = Path(img_path).relative_to(self.output_dir)
                html += f'<img src="{rel_path}" alt="{viz["title"]}">'
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def create_all_visualizations(self, df, eda_plan):
        """
        Create all visualizations from the EDA plan
        
        Args:
            df: Pandas DataFrame
            eda_plan: Dictionary with EDA plan
            
        Returns:
            list: List of visualization metadata
        """
        for i, viz_config in enumerate(eda_plan.get("visualizations", [])):
            # Add interactive flag for complex plots
            if viz_config["type"].lower() in ["scattermatrix", "heatmap", "correlation", "3d", 
                                             "parallel_coordinates", "treemap", "sunburst"]:
                viz_config["interactive"] = True
            
            self.create_visualization(df, viz_config)
        
        # Create a dashboard with all visualizations
        dashboard_path = self.get_visualization_dashboard()
        
        return self.visualization_results

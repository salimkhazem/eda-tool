import pandas as pd
import json
import base64
from pathlib import Path
import os
import zipfile
import io
from datetime import datetime

class ExportService:
    """Handles exporting EDA results in various formats."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def export_to_json(self, metadata, eda_plan, results):
        """
        Export all EDA results to a JSON file
        
        Args:
            metadata: Dictionary with dataset metadata
            eda_plan: Dictionary with the EDA plan
            results: Dictionary with EDA results
            
        Returns:
            str: Path to the exported JSON file
        """
        json_path = self.output_dir / 'eda_results.json'
        
        # Combine results into a single dictionary
        export_data = {
            'metadata': metadata,
            'eda_plan': eda_plan,
            'results': results,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Write to file
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(json_path)
    
    def export_to_zip(self, report_path, include_figures=True):
        """
        Export report and figures to a ZIP file
        
        Args:
            report_path: Path to the generated report
            include_figures: Whether to include visualization figures
            
        Returns:
            str: Path to the ZIP file
        """
        report_file = Path(report_path)
        figures_dir = report_file.parent / 'figures'
        
        # Create ZIP filename based on report name
        zip_path = self.output_dir / f'eda_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add the report file
            zipf.write(report_file, arcname=report_file.name)
            
            # Add figures if requested
            if include_figures and figures_dir.exists():
                for figure_file in figures_dir.glob('*.png'):
                    zipf.write(figure_file, arcname=f'figures/{figure_file.name}')
            
            # Add results JSON if it exists
            json_path = self.output_dir / 'eda_results.json'
            if json_path.exists():
                zipf.write(json_path, arcname=json_path.name)
        
        return str(zip_path)
    
    def get_base64_download_link(self, file_path, link_text):
        """
        Generate a base64 download link for a file
        
        Args:
            file_path: Path to the file
            link_text: Text for the download link
            
        Returns:
            str: HTML link for downloading the file
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        
        b64 = base64.b64encode(data).decode()
        file_name = Path(file_path).name
        mime_type = self._get_mime_type(file_path)
        
        href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}">{link_text}</a>'
        return href
    
    def _get_mime_type(self, file_path):
        """Get the MIME type based on file extension"""
        ext = Path(file_path).suffix.lower()
        
        mime_types = {
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.html': 'text/html',
            '.md': 'text/markdown',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.zip': 'application/zip'
        }
        
        return mime_types.get(ext, 'application/octet-stream')
    
    def export_dataset_with_insights(self, df, results):
        """
        Export the original dataset with added columns for insights
        
        Args:
            df: Original pandas DataFrame
            results: Dictionary with EDA results
            
        Returns:
            str: Path to the exported Excel file
        """
        enhanced_df = df.copy()
        
        # Add anomaly flags based on data quality checks
        for check_id, check in results['data_quality'].items():
            if check['type'].lower() == 'outliers':
                for col in check['columns']:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        # Extract bounds from results
                        results_data = check['results'].get(col, {})
                        lower_bound = results_data.get('lower_bound')
                        upper_bound = results_data.get('upper_bound')
                        
                        if lower_bound is not None and upper_bound is not None:
                            # Create anomaly flag column
                            flag_col = f'{col}_is_outlier'
                            enhanced_df[flag_col] = ((df[col] < lower_bound) | (df[col] > upper_bound))
        
        # Export to Excel
        excel_path = self.output_dir / 'dataset_with_insights.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            enhanced_df.to_excel(writer, sheet_name='Data with Insights', index=False)
            
            # Add a sheet with explanations
            explanations = pd.DataFrame({
                'Column': enhanced_df.columns,
                'Description': [
                    f"Outlier flag for {col.replace('_is_outlier', '')}" if col.endswith('_is_outlier') else 'Original data column'
                    for col in enhanced_df.columns
                ]
            })
            explanations.to_excel(writer, sheet_name='Column Explanations', index=False)
        
        return str(excel_path)

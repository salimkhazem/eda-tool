import argparse
import pandas as pd
from pathlib import Path
import sys
import json
import time
import traceback
import os
import shutil

from config import Config
from data_loader import DataLoader
from openai_service import OpenAIService
from eda_executor import EDAExecutor
from report_generator import ReportGenerator
from export_service import ExportService
from utils import export_results

def clean_output_directory(output_dir):
    """
    Clean the output directory to ensure fresh results
    
    Args:
        output_dir: Path to the output directory
    """
    output_path = Path(output_dir)
    
    # Clear figures directory
    figures_dir = output_path / "figures"
    if figures_dir.exists():
        # Remove all files in figures directory
        for file in figures_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error removing file {file}: {str(e)}")
    else:
        # Create the figures directory if it doesn't exist
        figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear interactive directory if it exists
    interactive_dir = output_path / "interactive"
    if interactive_dir.exists():
        for file in interactive_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error removing file {file}: {str(e)}")
    else:
        interactive_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove main output files
    for ext in ["md", "html", "json", "xlsx", "zip"]:
        for file in output_path.glob(f"*.{ext}"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Error removing file {file}: {str(e)}")

def run_eda(data_source, output_format='markdown', output_dir=None, export_formats=None):
    """
    Run the full EDA pipeline
    
    Args:
        data_source: Path to CSV/Excel file or Pandas DataFrame
        output_format: Format for the report ('markdown' or 'html')
        output_dir: Custom output directory path
        export_formats: List of additional export formats
        
    Returns:
        dict: Dict with paths to generated reports (html and markdown)
    """
    start_time = time.time()
    
    # Initialize configuration
    config = Config()
    if output_dir:
        config.output_dir = output_dir
    
    # Clean the output directory to ensure fresh results
    clean_output_directory(config.output_dir)
    
    # Initialize components
    data_loader = DataLoader(config)
    openai_service = OpenAIService(config)
    eda_executor = EDAExecutor(config)
    report_generator = ReportGenerator(config)
    export_service = ExportService(config)
    
    # Load and preprocess data
    print(f"🔍 Loading data from {data_source if isinstance(data_source, str) else 'DataFrame'}")
    try:
        df = data_loader.load_data(data_source)
        print(f"   ✓ Loaded data with shape: {df.shape}")
        
        # Optionally preprocess the data
        df_processed = data_loader.preprocess_data(df)
        if df_processed.shape != df.shape or (df_processed.dtypes != df.dtypes).any():
            print(f"   ✓ Applied preprocessing to the dataset")
            df = df_processed
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        raise
    
    # Extract metadata
    print(f"📊 Extracting metadata from dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    try:
        metadata = data_loader.extract_metadata(df)
        print(f"   ✓ Extracted metadata for {len(metadata['column_names'])} columns")
    except Exception as e:
        print(f"   ✗ Error extracting metadata: {e}")
        raise
    
    # Generate EDA plan with GPT
    print(f"🧠 Generating EDA plan with Azure OpenAI GPT-4o-mini")
    try:
        eda_plan = openai_service.get_eda_plan(metadata)
        print(f"   ✓ Generated EDA plan with {len(eda_plan.get('visualizations', []))} visualizations")
    except Exception as e:
        print(f"   ✗ Error generating EDA plan: {e}")
        print("   💡 Check your Azure OpenAI API key and endpoint in the .env file")
        raise
    
    # Execute EDA plan
    print(f"⚙️ Executing EDA plan")
    try:
        # Perform deep analysis automatically if EDA depth is set to deep
        perform_deep_analysis = config.eda_depth.lower() == 'deep'
        results = eda_executor.execute_plan(df, eda_plan, perform_deep_analysis=perform_deep_analysis)
        print(f"   ✓ Created {len(results['visualizations'])} visualizations")
        print(f"   ✓ Computed statistics for {len(results['statistics'])} columns")
        print(f"   ✓ Performed {len(results['data_quality'])} data quality checks")
        
        if perform_deep_analysis and 'deep_analysis' in results:
            print(f"   ✓ Completed deep analysis with {len(results['deep_analysis'])} components")
    except Exception as e:
        print(f"   ✗ Error executing EDA plan: {e}")
        raise
    
    # Generate reports - always generate both Markdown and HTML
    print(f"📝 Generating both Markdown and HTML reports")
    generated_reports = {}
    
    try:
        # Generate Markdown report first (always generated)
        markdown_path = report_generator.generate_markdown(df, metadata, eda_plan, results)
        generated_reports['markdown'] = markdown_path
        print(f"   ✓ Generated markdown report at: {markdown_path}")
        
        # Generate HTML report
        html_path = report_generator.generate_html(df, metadata, eda_plan, results)
        generated_reports['html'] = html_path
        print(f"   ✓ Generated HTML report at: {html_path}")
        
    except Exception as e:
        print(f"   ✗ Error generating reports: {e}")
        traceback.print_exc()
        # Ensure we have at least the markdown report
        if 'markdown' not in generated_reports:
            try:
                markdown_path = report_generator.generate_markdown(df, metadata, eda_plan, results)
                generated_reports['markdown'] = markdown_path
                print(f"   ✓ Generated markdown report at: {markdown_path}")
            except:
                raise
    
    # Export additional formats if requested
    if export_formats:
        print(f"📦 Exporting results in additional formats: {', '.join(export_formats)}")
        try:
            # Export to JSON (always done)
            json_path = export_service.export_to_json(metadata, eda_plan, results)
            print(f"   ✓ Exported JSON results to: {json_path}")
            
            # Export dataset with insights
            if 'excel' in export_formats:
                enhanced_dataset_path = export_service.export_dataset_with_insights(df, results)
                print(f"   ✓ Exported dataset with insights to: {enhanced_dataset_path}")
            
            # Create ZIP archive with all results
            primary_report = generated_reports.get('html') if 'html' in generated_reports else generated_reports.get('markdown')
            zip_path = export_service.export_to_zip(primary_report)
            print(f"   ✓ Created ZIP archive with all results: {zip_path}")
            
            # Use utility function for other formats
            export_paths = export_results(df, metadata, eda_plan, results, config.output_dir, export_formats)
            for fmt, path in export_paths.items():
                if fmt not in ['json', 'zip']:  # Already handled above
                    print(f"   ✓ Exported {fmt.upper()} to: {path}")
        except Exception as e:
            print(f"   ✗ Error exporting results: {e}")
            # Continue execution even if export fails
    
    elapsed_time = time.time() - start_time
    print(f"✅ EDA complete!")
    print(f"📄 Reports generated:")
    for fmt, path in generated_reports.items():
        print(f"  - {fmt.upper()}: {path}")
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
    
    # Return paths to all generated reports
    return generated_reports

def main():
    parser = argparse.ArgumentParser(description='Smart Automatic EDA with Azure OpenAI')
    parser.add_argument('data_source', type=str, help='Path to CSV or Excel file')
    parser.add_argument('--format', type=str, choices=['markdown', 'html'], default='html',
                      help='Primary output format (markdown or html) - both will be generated')
    parser.add_argument('--output-dir', type=str, help='Custom output directory')
    parser.add_argument('--depth', type=str, choices=['basic', 'intermediate', 'deep'], 
                      help='Analysis depth (overrides .env setting)')
    parser.add_argument('--export', type=str, nargs='+', choices=['json', 'csv', 'excel', 'zip'],
                      help='Additional export formats')
    
    args = parser.parse_args()
    
    try:
        # Set depth environment variable if specified
        if args.depth:
            import os
            os.environ["EDA_DEPTH"] = args.depth
            print(f"Setting analysis depth to: {args.depth}")
        
        run_eda(args.data_source, args.format, args.output_dir, args.export)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

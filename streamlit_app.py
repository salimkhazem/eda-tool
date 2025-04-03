import os
import tempfile
import base64
import logging
import io
import zipfile
import streamlit as st
import pandas as pd
import PIL
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increase PIL's maximum image size limit
# This should be done carefully as very large images can consume a lot of memory
PIL.Image.MAX_IMAGE_PIXELS = 4000000000  # Increase the limit

from main import run_eda
from config import Config 

st.set_page_config(
    page_title="Talan Smart EDA",
    page_icon="📊",
    layout="wide"  # Ensure wide layout is used 
)

def clear_output_directory():
    """Clear the output directory to ensure fresh results for each dataset"""
    config = Config()
    output_dir = Path(config.output_dir)
    
    if output_dir.exists():
        # Clear figures directory
        figures_dir = output_dir / "figures"
        if figures_dir.exists():
            try:
                for file in figures_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                logger.info(f"Cleared figures directory: {figures_dir}")
            except Exception as e:
                logger.error(f"Error clearing figures directory: {str(e)}")
        
        # Clear deep analysis directory
        deep_analysis_dir = output_dir / "deep_analysis"
        if deep_analysis_dir.exists():
            try:
                for file in deep_analysis_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                logger.info(f"Cleared deep analysis directory: {deep_analysis_dir}")
            except Exception as e:
                logger.error(f"Error clearing deep analysis directory: {str(e)}")
        
        # Clear main output files (markdown, html, etc.)
        for ext in ["md", "html", "json", "xlsx", "zip"]:
            for file in output_dir.glob(f"*.{ext}"):
                try:
                    file.unlink()
                    logger.info(f"Removed file: {file}")
                except Exception as e:
                    logger.error(f"Error removing file {file}: {str(e)}")

def main():
    # Create a layout with logo and title side by side
    col1, col2 = st.columns([1, 5])
    
    # Load and display the Talan logo in the first column
    logo_path = Path(__file__).parent / "talan_logo.png"
    if logo_path.exists():
        with col1:
            st.image(str(logo_path), width=120)
    else:
        # If logo doesn't exist, create a placeholder
        with col1:
            st.warning("Logo file not found")

    # Display the title in the second column with vertical alignment
    with col2:
        st.title("Smart Exploratory Data Analysis")
    
    st.markdown("""
    Upload your data and let Azure OpenAI's GPT-4o-mini generate a comprehensive EDA for you!
    
    This tool will:
    - Extract metadata from your dataset
    - Generate a tailored EDA plan using AI
    - Create visualizations and statistics
    - Perform data quality checks
    - Suggest feature engineering ideas
    - Generate an interactive report
    """)
    
    # Add a session state to track when a new file is uploaded or dataset is selected
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # Configuration options
    st.sidebar.header("Configuration")
    output_format = st.sidebar.selectbox("Primary Report Format", ["markdown", "html"], index=1)
    eda_depth = st.sidebar.selectbox("Analysis Depth", ["basic", "intermediate", "deep"], index=1)
    
    # Add a "Clear Previous Results" button
    if st.sidebar.button("Clear Previous Results"):
        clear_output_directory()
        st.sidebar.success("Previous results cleared!")
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        custom_output_dir = st.text_input("Custom Output Directory", "")
        max_rows_sample = st.number_input("Sample Rows to Show", min_value=1, max_value=20, value=5)
    
    # Set environment variables based on UI selections
    os.environ["EDA_DEPTH"] = eda_depth
    os.environ["MAX_ROWS_SAMPLE"] = str(max_rows_sample)
    
    # Example datasets
    with st.sidebar.expander("Example Datasets"):
        example_option = st.selectbox(
            "Select an example dataset",
            ["None", "Iris Dataset", "Titanic Dataset", "Housing Dataset"]
        )
    
    # Check if dataset selection has changed
    dataset_changed = False
    current_dataset = f"example_{example_option}" if example_option != "None" else (uploaded_file.name if uploaded_file else None)
    
    if current_dataset != st.session_state.current_dataset:
        dataset_changed = True
        st.session_state.current_dataset = current_dataset
        # Clear previous output if dataset has changed
        if dataset_changed:
            clear_output_directory()
    
    if example_option != "None":
        # Load example dataset
        if example_option == "Iris Dataset":
            try:
                from sklearn.datasets import load_iris
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['species'] = pd.Series(data.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                st.success("Loaded Iris Dataset")
                show_dataframe = True
            except ImportError:
                st.warning("scikit-learn not installed. Install it with: pip install scikit-learn")
                show_dataframe = False
                
        elif example_option == "Titanic Dataset":
            df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
            st.success("Loaded Titanic Dataset")
            show_dataframe = True
            
        elif example_option == "Housing Dataset":
            try:
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.success("Loaded California Housing Dataset")
                show_dataframe = True
            except ImportError:
                st.warning("scikit-learn not installed. Install it with: pip install scikit-learn")
                show_dataframe = False
                
        if show_dataframe:
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Run EDA on Example Dataset"):
                with st.spinner("Generating EDA... This may take a minute..."):
                    # Save dataset to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        temp_path = tmp_file.name
                        
                    try:
                        # Run EDA
                        output_dir = custom_output_dir if custom_output_dir else None
                        reports = run_eda(temp_path, output_format, output_dir, export_formats=None)
                        
                        # Display results
                        _display_results(reports["markdown"], output_format, reports)
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
    elif uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Show data preview
            if Path(uploaded_file.name).suffix.lower() == '.csv':
                df = pd.read_csv(temp_path)
            else:  # Excel
                df = pd.read_excel(temp_path)
                
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                # Run EDA button
                if st.button("Run EDA"):
                    with st.spinner("Generating EDA... This may take a minute..."):
                        output_dir = custom_output_dir if custom_output_dir else None
                        reports = run_eda(temp_path, output_format, output_dir, export_formats=None)
                        
                        # Display results
                        _display_results(reports["markdown"], output_format, reports)
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

def _display_results(report_path, output_format, all_reports=None):
    """Helper function to display EDA results"""
    st.success(f"EDA complete! Reports generated in both Markdown and HTML formats")
    
    # Read the primary report file
    with open(report_path, 'r') as f:
        primary_report_content = f.read()
    
    # Apply custom CSS to make tabs take full width of the page
    st.markdown("""
    <style>
    /* Make tabs take the full width of the page */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        flex-grow: 1;
        flex-basis: 0;
        font-size: 1.1em;
        flex-shrink: 1;
        justify-content: center;
        min-width: auto;
    }
    
    /* Make sure the tab content takes full page width too */
    .stTabs [data-baseweb="tab-panel"] {
        width: 100%;
    }
    
    /* Container also takes full width */
    .block-container, .stTabs {
        width: 100%;
        max-width: 100%;
        padding: 0;
    }
    
    /* Ensure content is wide too */
    div[data-testid="stVerticalBlock"] {
        width: 100%;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views with full width
    tab1, tab2, tab3, tab4 = st.tabs(["HTML Report", "Markdown Report", "Visualizations", "Download"])
    
    with tab1:
        # Display the HTML report - use full width for the iframe with improved styling
        if 'html' in all_reports:
            with open(all_reports['html'], 'r') as f:
                html_content = f.read()
            
            # Create container that uses full width
            container = st.container()
            with container:
                try:
                    # Fix the error by using a safer HTML escaping approach
                    # Replace quotes and escape problematic characters properly
                    escaped_html = (html_content
                                    .replace('&', '&amp;')
                                    .replace('<', '&lt;')
                                    .replace('>', '&gt;')
                                    .replace('"', '&quot;')
                                    .replace("'", '&#39;')
                                    .replace('\n', '&#10;')
                                    .replace('\r', '&#13;'))
                    
                    # Create a simpler iframe HTML that is less prone to escaping issues
                    iframe_html = f"""
                    <div style="width:100%; height:800px; overflow:hidden;">
                        <iframe srcdoc="{escaped_html}" style="width:100%; height:100%; border:none;"></iframe>
                    </div>
                    """
                    
                    st.components.v1.html(iframe_html, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error displaying HTML report: {str(e)}")
                    st.info("Displaying HTML report as raw text instead")
                    # Fallback to displaying as text
                    st.code(html_content[:2000] + "..." if len(html_content) > 2000 else html_content, language="html")
        else:
            st.warning("HTML report was not generated.")
    
    with tab2:
        # Display the markdown report - already using full width
        if 'markdown' in all_reports:
            # First, collect all image files to display them directly
            output_dir = Path(report_path).parent
            figures_dir = output_dir / "figures"
            deep_analysis_dir = output_dir / "deep_analysis"
            
            # Get all image files
            image_files = []
            if figures_dir.exists():
                image_files.extend(list(figures_dir.glob("*.png")))
            if deep_analysis_dir.exists():
                image_files.extend(list(deep_analysis_dir.glob("*.png")))
            
            # Create a mapping from filename to actual image data
            images = {}
            for img_path in image_files:
                try:
                    with open(img_path, "rb") as img_file:
                        # Create base64 encoded image data
                        import base64
                        encoded = base64.b64encode(img_file.read()).decode()
                        # Store with just the filename as key
                        images[img_path.name] = encoded
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            
            # Read markdown content
            with open(all_reports['markdown'], 'r') as f:
                md_content = f.read()
            
            # Replace image references in markdown with data URI images
            import re
            
            # Pattern to find markdown image syntax: ![alt text](path/to/image.png)
            img_pattern = r'!\[(.*?)\]\((.*?)\)'
            
            def replace_with_data_uri(match):
                alt_text = match.group(1)
                img_path = match.group(2)
                
                # Extract just the filename
                img_filename = Path(img_path).name
                
                # Check if we have this image in our mapping
                if img_filename in images:
                    return f'![{alt_text}](data:image/png;base64,{images[img_filename]})'
                else:
                    # If image not found, return original reference
                    return match.group(0)
            
            # Replace all image references with data URIs
            md_content = re.sub(img_pattern, replace_with_data_uri, md_content)
            
            # Display modified markdown
            st.markdown(md_content, unsafe_allow_html=True)
        else:
            st.warning("Markdown report was not generated.")
    
    with tab3:
        # Show visualization gallery with improved layout
        st.subheader("Visualization Gallery")
        
        # First, collect all image files including those in subdirectories
        figures_dir = Path(report_path).parent
        all_image_files = []
        
        # Check main figures directory
        main_figures_dir = figures_dir / "figures"
        if main_figures_dir.exists():
            all_image_files.extend(list(main_figures_dir.glob("*.png")))
        
        # Check for deep analysis figures
        deep_analysis_dir = figures_dir / "deep_analysis"
        if deep_analysis_dir.exists():
            all_image_files.extend(list(deep_analysis_dir.glob("*.png")))
        
        # Check for any other PNG files directly in output dir
        all_image_files.extend(list(figures_dir.glob("*.png")))
        
        # Sort images by modification time (newest first)
        all_image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if all_image_files:
            # Create filter for image types
            image_types = set()
            for img_path in all_image_files:
                # Extract image type from filename (e.g., "histogram", "scatter", etc.)
                img_type = img_path.stem.split('_')[-1] if '_' in img_path.stem else "other"
                image_types.add(img_type)
            
            
            
            filtered_images = all_image_files
            
            # Create a responsive grid system using st.columns
            num_columns = 3  # Default to 3 columns on larger screens
            
            # Create a gallery of visualizations with captions
            for i in range(0, len(filtered_images), num_columns):
                cols = st.columns(num_columns)
                
                # Add images to columns
                for j in range(num_columns):
                    idx = i + j
                    if idx < len(filtered_images):
                        img_path = filtered_images[idx]
                        try:
                            # Get a descriptive caption from the filename
                            caption = img_path.stem.replace('_', ' ').title()
                            
                            # Display the image with better styling
                            with cols[j]:
                                st.image(
                                    str(img_path), 
                                    caption=caption,
                                    use_container_width=True
                                )
                                
                        except Exception as e:
                            cols[j].error(f"Error displaying image {img_path.name}: {str(e)}")
            
            
        else:
            st.info("No visualizations were generated.")
    
    with tab4:
        # Download options - Use full width columns for better layout
        st.subheader("Download Reports")
        
        # Create two columns for report downloads to use space efficiently
        col1, col2 = st.columns(2)
        
        # Markdown report download
        if 'markdown' in all_reports:
            with open(all_reports['markdown'], 'r') as f:
                md_content = f.read()
            md_b64 = base64.b64encode(md_content.encode()).decode()
            col1.markdown(f'<a href="data:text/markdown;base64,{md_b64}" download="eda_report.md" class="button" style="display:inline-block; padding:10px 15px; background-color:#4CAF50; color:white; text-decoration:none; border-radius:4px; margin:10px 0; width:100%; text-align:center;">Download Markdown Report</a>', unsafe_allow_html=True)
        
        # HTML report download
        if 'html' in all_reports:
            with open(all_reports['html'], 'r') as f:
                html_content = f.read()
            html_b64 = base64.b64encode(html_content.encode()).decode()
            col2.markdown(f'<a href="data:text/html;base64,{html_b64}" download="eda_report.html" class="button" style="display:inline-block; padding:10px 15px; background-color:#2196F3; color:white; text-decoration:none; border-radius:4px; margin:10px 0; width:100%; text-align:center;">Download HTML Report</a>', unsafe_allow_html=True)
        
        # Add download all visualizations as zip
        if all_image_files:
            st.markdown("### Download All Visualizations")
            
            # Create a zip file of all visualizations
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for img_path in all_image_files:
                    zip_file.write(img_path, arcname=img_path.name)
            
            zip_buffer.seek(0)
            zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode()
            
            st.markdown(f'<a href="data:application/zip;base64,{zip_b64}" download="visualizations.zip" class="button" style="display:inline-block; padding:10px 15px; background-color:#FF9800; color:white; text-decoration:none; border-radius:4px; margin:10px 0; width:100%; text-align:center;">Download All Visualizations (ZIP)</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

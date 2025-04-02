import streamlit as st
import pandas as pd
import os
from pathlib import Path
import tempfile
import base64
import time
import glob
import PIL
from PIL import Image
import io
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increase PIL's maximum image size limit
# This should be done carefully as very large images can consume a lot of memory
PIL.Image.MAX_IMAGE_PIXELS = 400000000  # Increase the limit

from main import run_eda
from config import Config

st.set_page_config(
    page_title="Smart EDA with GPT-4o-mini",
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
            st.dataframe(df.head(), use_container_width=True)  # Use full width
            
            if st.button("Run EDA on Example Dataset"):
                with st.spinner("Generating EDA... This may take a minute..."):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        temp_path = tmp_file.name
                        
                    try:
                        # Run EDA
                        output_dir = custom_output_dir if custom_output_dir else None
                        reports = run_eda(temp_path, output_format, output_dir, export_formats=None)
                        _display_results(reports.get('html', reports.get('markdown')), output_format, reports)
                    finally:
                        # Clean up
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
            st.dataframe(df.head(), use_container_width=True)  # Use full width
            
            col1, col2 = st.columns([1, 3])
            with col1:
                # Run EDA button
                if st.button("Run EDA"):
                    with st.spinner("Generating EDA... This may take a minute..."):
                        start_time = time.time()
                        
                        # Setup progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("Loading data...")
                        progress_bar.progress(10)
                        time.sleep(0.5)
                        
                        # Run the EDA
                        status_text.text("Generating EDA plan with gpt-4o-mini...")
                        progress_bar.progress(30)
                        
                        output_dir = custom_output_dir if custom_output_dir else None
                        reports = run_eda(temp_path, output_format, output_dir, export_formats=None)
                        
                        status_text.text("Creating visualizations...")
                        progress_bar.progress(70)
                        time.sleep(0.5)
                        
                        status_text.text("Generating report...")
                        progress_bar.progress(90)
                        time.sleep(0.5)
                        
                        # Complete
                        elapsed_time = time.time() - start_time
                        status_text.text(f"Completed in {elapsed_time:.2f} seconds")
                        progress_bar.progress(100)
                        
                        _display_results(reports.get('html', reports.get('markdown')), output_format, reports)
        
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
            with open(all_reports['markdown'], 'r') as f:
                md_content = f.read()
            st.markdown(md_content)
        else:
            st.warning("Markdown report was not generated.")
    
    with tab3:
        # Show visualization gallery with improved layout
        st.subheader("Visualization Gallery")
        figures_dir = Path(report_path).parent / "figures"
        if figures_dir.exists():
            image_files = list(figures_dir.glob("*.png"))
            if image_files:
                # Add a full-width container for visualizations
                full_width_container = st.container()
                with full_width_container:
                    # Create a more responsive grid system using st.columns
                    num_columns = 3  # Default to 3 columns on larger screens
                    
                    # Group images into rows
                    for i in range(0, len(image_files), num_columns):
                        cols = st.columns(num_columns)
                        for j in range(num_columns):
                            if i + j < len(image_files):
                                try:
                                    img_path = image_files[i + j]
                                    img_name = img_path.name
                                    
                                    # Open and check image size first before displaying
                                    try:
                                        with Image.open(img_path) as img:
                                            # Check if image is unreasonably large
                                            width, height = img.size
                                            if width * height > 50000000:  # If image is very large
                                                # Calculate new dimensions while preserving aspect ratio
                                                ratio = width / height
                                                if width > height:
                                                    new_width = 2000  # Max width
                                                    new_height = int(new_width / ratio)
                                                else:
                                                    new_height = 2000  # Max height
                                                    new_width = int(new_height * ratio)
                                                
                                                # Resize image
                                                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                                
                                                # Save to temporary buffer
                                                img_buffer = io.BytesIO()
                                                resized_img.save(img_buffer, format="PNG")
                                                img_buffer.seek(0)
                                                
                                                # Display the resized image
                                                cols[j].image(img_buffer, caption=f"{img_name} (resized)", use_container_width=True)
                                            else:
                                                # Image is reasonable size, display normally
                                                # Use container_width=True to ensure the image uses the full column width
                                                cols[j].image(str(img_path), caption=img_name, use_container_width=True)
                                    except PIL.Image.DecompressionBombError:
                                        # If we still get the error, show a placeholder instead
                                        cols[j].warning(f"Image too large to display: {img_name}")
                                except Exception as e:
                                    cols[j].error(f"Error displaying image {img_name}: {str(e)}")
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
        
        # Visualizations download - Use grid layout for better use of space
        st.markdown("### Download Individual Visualizations")
        figures_dir = Path(report_path).parent / "figures"
        if figures_dir.exists():
            image_files = list(figures_dir.glob("*.png"))
            if image_files:
                # Create a gallery of download links, 3 per row
                for i in range(0, len(image_files), 6):
                    cols = st.columns(6)
                    for j in range(6):
                        if i + j < len(image_files):
                            try:
                                img_path = image_files[i + j]
                                img_name = img_path.name
                                
                                # Check if image is too large before preparing download
                                try:
                                    with Image.open(img_path) as img:
                                        # Just check if we can open it - PIL will raise DecompressionBombError if too large
                                        pass
                                        
                                    # If we get here, image is OK to provide for download
                                    with open(img_path, "rb") as file:
                                        img_bytes = file.read()
                                        b64 = base64.b64encode(img_bytes).decode()
                                        # Make buttons more consistent and full-width
                                        href = f'<a href="data:image/png;base64,{b64}" download="{img_path.name}" style="display:block; margin:10px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; background-color:#f0f0f0; padding:8px 12px; border-radius:4px; text-align:center; color:#333; text-decoration:none; width:100%;">{img_path.name}</a>'
                                        cols[j].markdown(href, unsafe_allow_html=True)
                                        
                                except PIL.Image.DecompressionBombError:
                                    cols[j].warning(f"Image too large to download: {img_path.name}")
                                    
                            except Exception as e:
                                cols[j].error(f"Error with {img_path.name}: {str(e)}")

if __name__ == "__main__":
    main()

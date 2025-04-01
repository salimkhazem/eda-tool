# Smart EDA with GPT-4o-mini

This tool performs automated Exploratory Data Analysis (EDA) using OpenAI's GPT-4o-mini model. It analyzes datasets, creates visualizations, performs statistical analysis, and generates comprehensive reports.

## Features

- Supports CSV and Excel files as input
- Extracts dataset metadata (columns, types, samples)
- Uses GPT-4o-mini to create a tailored EDA plan
- Executes statistical analysis with pandas
- Creates visualizations with matplotlib and seaborn
- Performs data quality checks
- Suggests feature engineering ideas
- Generates interactive reports (Markdown, HTML)
- Provides both CLI and Streamlit interface

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your API details in the `.env` file using the `.env.example` template

### Setting Up PDF Support

To generate PDF reports, you'll need additional dependencies:

```bash
# Run the helper script to install PDF libraries
python install_pdf_dependencies.py
```

Alternatively, install them manually:

1. **WeasyPrint** (recommended):
   ```bash
   pip install weasyprint
   ```

2. **PDFKit** (alternative, requires wkhtmltopdf):
   ```bash
   pip install pdfkit
   ```
   You'll also need to install wkhtmltopdf:
   - macOS: `brew install wkhtmltopdf`
   - Debian/Ubuntu: `sudo apt-get install wkhtmltopdf`
   - Windows: Download installer from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html)

3. **markdown2pdf** (fallback):
   ```bash
   pip install markdown2pdf
   ```

## Usage

### Command Line Interface

```bash
python main.py path/to/your/data.csv
```

It will generate all reports format

```bash
python main.py path/to/your/data.csv --format html --output-dir custom_output
```

Options:
- `data_source`: Path to CSV or Excel file (required)
- `--format`: Output format (`markdown`, `html`, or `pdf`, default: `markdown`)
- `--output-dir`: Custom output directory (optional)

### Streamlit Interface

```bash
streamlit run streamlit_app.py
```

Then open your browser at http://localhost:8501

## Configuration

Edit the `.env` file to customize:

- Azure OpenAI API credentials
- Default output directory
- EDA depth (basic, intermediate, deep)
- Maximum sample rows

## Output

The tool generates:
- A comprehensive EDA report in Markdown, HTML or PDF format
- Visualizations saved in the figures directory
- Statistical summaries
- Data quality analysis
- Feature engineering suggestions

## Requirements

- Python 3.8+
- OpenAI Python SDK
- pandas
- matplotlib
- seaborn
- streamlit (for web interface)
- Additional libraries for PDF support

# Advanced Predictive Value Calculator

This interactive web application, built with Plotly Dash, provides a professional tool for analyzing and visualizing the performance of diagnostic tests. It allows users to explore how a test's Positive Predictive Value (PPV) and Negative Predictive Value (NPV) change across different disease prevalence rates.

## Features

- **Interactive Calculator:** Input a test's sensitivity and specificity to instantly see how PPV and NPV change with prevalence.
- **Dynamic Visualizations:** View the relationship between prevalence and predictive values as a line chart, area chart, or likelihood ratio plot.
- **Clinical Scenarios:** Explore pre-configured scenarios for common diagnostic tests like COVID-19 rapid tests, mammography, and more.
- **Comparison Mode:** Compare multiple tests side-by-side to evaluate their relative performance.
- **Educational Case Studies:** Learn about key concepts like the screening paradox and the effect of high prevalence on test accuracy.
- **Dark/Light Theme:** Toggle between themes for comfortable viewing.
- **PDF Export:** Generate a PDF summary of your analysis.

## Dependencies

- Python 3
- Dash
- Plotly
- NumPy
- ReportLab (for PDF export)

All dependencies are listed in the `requirements.txt` file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/derekpadilla/ppvnpv.git
    cd ppvnpv
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Dash application with the following command:

```bash
python display-graph.py
```

Then, open your web browser and navigate to `http://127.0.0.1:8050/` to use the application.
Positive and Negative Predictive Value Calculator (PredictiveValue.info)

Overview:
---------
This is a web application that provides an interactive tool for users to calculate Positive Predictive Value (PPV) and Negative Predictive Value (NPV) based on user-provided sensitivity and specificity values.

Features:
---------
1. Users can input the sensitivity and specificity values for their test.
2. A dynamic graph displaying the variation of PPV and NPV with prevalence.
3. Definitions and formulas for PPV, NPV, and Prevalence.
4. An illustrative example demonstrating the impact of sensitivity, specificity, and prevalence on PPV and NPV.
5. External links for a detailed discussion on PPV and NPV and credits for the tools used in the development.

Requirements:
-------------
- Python
- Dash by Plotly
- NumPy

Running the Application:
------------------------
1. Ensure you have Python installed on your system.
2. Install the necessary libraries by using the command: 
   ```
   pip install dash numpy
   ```
3. Navigate to the directory containing 'main.py' in your terminal.
4. Run the script using the command:
   ```
   python main.py
   ```

The web application will be hosted on a local server, typically `http://127.0.0.1:8050/` (or another port if 8050 is busy). Open this link in your web browser to use the tool.

Additional Resources:
---------------------
- [Article on Medium discussing PPV and NPV using COVID-19 antibody tests as an example.](https://medium.com/@liketortilla/can-you-trust-your-antibody-test-results-89e438b9bd4c)
- [More about the Dash Python framework by Plotly](https://plotly.com/dash/)

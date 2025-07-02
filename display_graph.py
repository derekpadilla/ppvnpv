import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors as rl_colors

# Initialize the Dash app with Bootstrap components and a responsive viewport meta tag
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
                title='PredictiveValue.info')
server = app.server

# --- Data Definitions ---

CLINICAL_SCENARIOS = {
    'covid_rapid': {'name': 'COVID-19 Rapid Test', 'sensitivity': 95.0, 'specificity': 99.0, 'typical_prevalence': [0.5, 2, 5, 10], 'description': 'Rapid antigen test for SARS-CoV-2 detection', 'notes': 'Performance varies by viral load and symptom duration'},
    'mammography': {'name': 'Mammography Screening', 'sensitivity': 85.0, 'specificity': 95.0, 'typical_prevalence': [0.3, 0.5, 0.8], 'description': 'Annual mammography for breast cancer screening', 'notes': 'Sensitivity lower in dense breast tissue'},
    'psa_screening': {'name': 'PSA Screening', 'sensitivity': 90.0, 'specificity': 75.0, 'typical_prevalence': [10, 15, 25], 'description': 'Prostate-specific antigen for prostate cancer', 'notes': 'High false positive rate, age-dependent'},
    'troponin': {'name': 'Cardiac Troponin', 'sensitivity': 98.0, 'specificity': 92.0, 'typical_prevalence': [5, 15, 30], 'description': 'High-sensitivity troponin for MI diagnosis', 'notes': 'Time-dependent sensitivity after symptom onset'},
    'hiv_elisa': {'name': 'HIV ELISA', 'sensitivity': 99.5, 'specificity': 99.8, 'typical_prevalence': [0.1, 0.4, 1.2], 'description': 'Enzyme-linked immunosorbent assay for HIV', 'notes': 'Confirmatory testing required for positives'},
    'colonoscopy': {'name': 'Colonoscopy Screening', 'sensitivity': 95.0, 'specificity': 90.0, 'typical_prevalence': [0.5, 1.0, 2.0], 'description': 'Colonoscopy for colorectal cancer screening', 'notes': 'Gold standard but invasive procedure'}
}

CASE_STUDIES = {
    'screening_paradox': {'title': 'The Screening Paradox', 'scenario': 'A new cancer screening test has 99% sensitivity and 95% specificity. Should it be used for population screening?', 'test_params': {'sensitivity': 99, 'specificity': 95}, 'prevalence': 0.1, 'learning_points': ['Even excellent tests can have poor PPV in low-prevalence populations', 'This test would have only 2% PPV at 0.1% prevalence', 'Most positive results would be false positives', 'Consider psychological and economic costs of false positives']},
    'high_prevalence': {'title': 'Emergency Department Diagnosis', 'scenario': 'In an emergency department, 30% of chest pain patients have MI. How does this high prevalence affect troponin testing?', 'test_params': {'sensitivity': 98, 'specificity': 92}, 'prevalence': 30, 'learning_points': ['High prevalence dramatically improves PPV', 'PPV increases from 12% (at 1% prevalence) to 84% (at 30%)', 'NPV remains excellent (99.9%) due to high sensitivity', 'Clinical context is crucial for test interpretation']}
}

# --- App Layout ---

app.layout = dbc.Container(id='main-container', fluid=True, children=[
    dcc.Store(id='app-theme', data='light'),
    dcc.Store(id='comparison-tests', data=[]),

    dbc.Row([
        dbc.Col(
            dbc.Button([
                html.I(id='theme-icon', className="fas fa-moon"), html.Span(id='theme-text', children='Dark Mode')
            ], id='theme-toggle', className='theme-toggle-btn'),
            width='auto',
            className='d-flex justify-content-end p-3'
        )
    ], justify='end'),

    dbc.Row([
        dbc.Col(
            html.Div(id='header', children=[
                html.H1([html.I(className="fas fa-calculator"), " Advanced Predictive Value Calculator"]),
                html.P("Professional diagnostic test analysis with interactive visualization and clinical scenarios")
            ]),
            width=12,
            className='text-center'
        )
    ]),

    html.Div(className='main-content', children=[
        dcc.Tabs(id='mode-tabs', value='calculator', children=[
            dcc.Tab(label='Calculator', value='calculator', className='custom-tab'),
            dcc.Tab(label='Clinical Scenarios', value='scenarios', className='custom-tab'),
            dcc.Tab(label='Comparison Mode', value='comparison', className='custom-tab'),
            dcc.Tab(label='Case Studies', value='cases', className='custom-tab'),
        ]),
        html.Div(id='tab-content', className='p-3')
    ]),

    html.Footer(id='footer', className='p-3 text-center', children=[
        html.P([
            "Enhanced by AI • Created by ",
            html.A("Derek Padilla", href="https://derekpadilla.com", target="_blank"),
            " • Built with ",
            html.A("Plotly Dash", href="https://plotly.com/dash/", target="_blank")
        ])
    ])
])

# --- Layout Helper Functions for Tabs ---

def create_calculator_tab():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.H3([html.I(className="fas fa-sliders-h"), " Test Parameters"]),
                dbc.Row([
                    dbc.Col([
                        html.Label([html.I(className="fas fa-search-plus"), " Sensitivity (%)"]),
                        dbc.Input(id='sens', value=98, type='number', min=0, max=100, step=0.1),
                        html.Small("Percentage of true positives correctly identified", className='text-muted')
                    ], width=12, md=6),
                    dbc.Col([
                        html.Label([html.I(className="fas fa-search-minus"), " Specificity (%)"]),
                        dbc.Input(id='spec', value=85, type='number', min=0, max=100, step=0.1),
                        html.Small("Percentage of true negatives correctly identified", className='text-muted')
                    ], width=12, md=6)
                ], className='mb-3'),
                html.Label([html.I(className="fas fa-percentage"), " Focus Prevalence (%)"]),
                dcc.Slider(id='focus-prevalence', min=0.1, max=50, step=0.1, value=5, marks={i: f'{i}%' for i in [0.1, 1, 5, 10, 20, 50]}, tooltip={'placement': 'bottom', 'always_visible': True}),
                html.Small("Drag to see PPV/NPV at a specific prevalence", className='text-muted d-block mt-2')
            ]),
            className='mb-4'
        ),
        html.Div(id='realtime-results'),
        dbc.Card(
            dbc.CardBody([
                html.H3([html.I(className="fas fa-chart-line"), " Interactive Predictive Values"]),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id='graph-type',
                            options=['Line Chart', 'Area Chart', 'Likelihood Ratios'],
                            value='Line Chart',
                            clearable=False
                        ), width=12, md=6
                    ),
                    dbc.Col(
                        dbc.Button([html.I(className="fas fa-download"), " Export Report"], id='export-btn', color="primary", className="w-100"),
                        width=12, md=6, className="mt-3 mt-md-0"
                    ),
                ]),
                dcc.Download(id="download-pdf-report"),
                html.Div(id='output-graph')
            ]),
            className='mb-4'
        ),
        dbc.Row([
            dbc.Col(html.Div(id='results-summary'), width=12, md=6),
            dbc.Col(html.Div(id='clinical-insights'), width=12, md=6, className="mt-4 mt-md-0")
        ])
    ])

def create_scenarios_tab():
    return dbc.Card(
        dbc.CardBody([
            html.H3([html.I(className="fas fa-user-md"), " Clinical Scenarios"]),
            html.P("Select a pre-configured clinical scenario to explore real-world diagnostic test performance."),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(scenario['name'], className='card-title'),
                            html.P(scenario['description'], className='card-text'),
                            html.Div([
                                dbc.Badge(f"Sens: {scenario['sensitivity']}%", color="success", className='me-2'),
                                dbc.Badge(f"Spec: {scenario['specificity']}%", color="info")
                            ]),
                            dbc.Button([html.I(className="fas fa-play"), " Load Scenario"], id={'type': 'scenario-btn', 'index': key}, color="primary", className='mt-3')
                        ])
                    ]),
                    width=12,
                    md=6,
                    lg=4,
                    className='mb-4'
                ) for key, scenario in CLINICAL_SCENARIOS.items()
            ])
        ])
    )

def create_comparison_tab():
    return dbc.Card(
        dbc.CardBody([
            html.H3([html.I(className="fas fa-balance-scale"), " Test Comparison"]),
            html.P("Compare multiple diagnostic tests side-by-side to evaluate their relative performance."),
            dbc.Row([
                dbc.Col(dcc.Input(id='test-name', placeholder='Test Name', className='mb-2 mb-md-0'), width=12, md=3),
                dbc.Col(dcc.Input(id='test-sens', placeholder='Sensitivity %', type='number', min=0, max=100, step=0.1, className='mb-2 mb-md-0'), width=12, md=3),
                dbc.Col(dcc.Input(id='test-spec', placeholder='Specificity %', type='number', min=0, max=100, step=0.1, className='mb-2 mb-md-0'), width=12, md=3),
                dbc.Col(dbc.Button([html.I(className="fas fa-plus"), " Add Test"], id='add-test-btn', color="primary", className='w-100'), width=12, md=3)
            ]),
            html.Div(id='comparison-results', className='mt-4')
        ])
    )

def create_cases_tab():
    return dbc.Card(
        dbc.CardBody([
            html.H3([html.I(className="fas fa-graduation-cap"), " Educational Case Studies"]),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(case['title'], className='card-title'),
                            html.P(case['scenario'], className='card-text'),
                            html.H5("Learning Points:"),
                            html.Ul([html.Li(point) for point in case['learning_points']]),
                            dbc.Button([html.I(className="fas fa-microscope"), " Explore Case"], id={'type': 'case-btn', 'index': key}, color="primary", className='mt-3')
                        ])
                    ]),
                    width=12,
                    md=6,
                    className='mb-4'
                ) for key, case in CASE_STUDIES.items()
            ])
        ])
    )


# --- Calculation and Component Generation ---

def calculate_predictive_values(sens, spec, prev):
    sens_prop = np.array(sens) / 100
    spec_prop = np.array(spec) / 100
    prev_prop = np.array(prev) / 100

    with np.errstate(divide='ignore', invalid='ignore'):
        ppv = 100 * np.nan_to_num((sens_prop * prev_prop) / ((sens_prop * prev_prop) + ((1 - spec_prop) * (1 - prev_prop))))
        npv = 100 * np.nan_to_num((spec_prop * (1 - prev_prop)) / (((1 - sens_prop) * prev_prop) + (spec_prop * (1 - prev_prop))))
    return ppv, npv

def calculate_likelihood_ratios(sens, spec):
    sens_prop = sens / 100
    spec_prop = spec / 100
    with np.errstate(divide='ignore'):
        lr_pos = sens_prop / (1 - spec_prop) if (1 - spec_prop) != 0 else float('inf')
        lr_neg = (1 - sens_prop) / spec_prop if spec_prop != 0 else float('inf')
    return lr_pos, lr_neg

def create_main_graph(graph_type, sens, spec, focus_prev, theme):
    prev_range = np.linspace(0.1, 50, 200)
    ppv, npv = calculate_predictive_values(sens, spec, prev_range)
    lr_pos, lr_neg = calculate_likelihood_ratios(sens, spec)

    fig = go.Figure()
    plot_bgcolor = '#2d2d2d' if theme == 'dark' else '#FFFFFF'
    paper_bgcolor = '#2d2d2d' if theme == 'dark' else '#FFFFFF'
    font_color = '#e0e0e0' if theme == 'dark' else '#2C3E50'
    grid_color = '#404040' if theme == 'dark' else '#E9ECEF'

    if graph_type != 'Likelihood Ratios':
        trace_ppv = go.Scatter(x=prev_range, y=ppv, mode='lines', name='PPV', line={'color': '#28a745', 'width': 4}, hovertemplate='<b>PPV</b><br>Prevalence: %{x:.1f}%<br>Value: %{y:.1f}%<extra></extra>')
        trace_npv = go.Scatter(x=prev_range, y=npv, mode='lines', name='NPV', line={'color': '#17a2b8', 'width': 4}, hovertemplate='<b>NPV</b><br>Prevalence: %{x:.1f}%<br>Value: %{y:.1f}%<extra></extra>')
        if graph_type == 'Area Chart':
            trace_ppv.fill = 'tonexty'
            trace_npv.fill = 'tozeroy'
        fig.add_traces([trace_ppv, trace_npv])
        fig.add_vline(x=focus_prev, line_dash="dash", line_color='#007bff', annotation_text=f"Focus: {focus_prev:.1f}%", annotation_position="top right")
    else:
        fig.add_trace(go.Scatter(x=[0.1, 50], y=[lr_pos, lr_pos], mode='lines', name='LR+', line={'color': '#28a745', 'width': 4, 'dash': 'dash'}, hovertemplate=f'<b>LR+</b>: {lr_pos:.2f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=[0.1, 50], y=[lr_neg, lr_neg], mode='lines', name='LR-', line={'color': '#17a2b8', 'width': 4, 'dash': 'dash'}, hovertemplate=f'<b>LR-</b>: {lr_neg:.2f}<extra></extra>'))
        fig.update_yaxes(type="log")

    fig.update_layout(
        xaxis_title="Disease Prevalence (%)",
        yaxis_title="Predictive Value (%)" if graph_type != 'Likelihood Ratios' else "Likelihood Ratio (log scale)",
        hovermode='x unified', height=400, showlegend=True,
        plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor,
        font={'color': font_color},
        xaxis={'gridcolor': grid_color}, yaxis={'gridcolor': grid_color},
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})

def generate_realtime_results(sens, spec, focus_prev):
    focus_ppv, focus_npv = calculate_predictive_values(sens, spec, focus_prev)
    return dbc.Card(
        dbc.CardBody([
            html.H4([html.I(className="fas fa-crosshairs"), f" Results at {focus_prev:.1f}% Prevalence"], className='text-center'),
            dbc.Row([
                dbc.Col([html.P("Positive Predictive Value (PPV)"), html.H2(f"{focus_ppv:.1f}%", className='text-success')], className='text-center'),
                dbc.Col([html.P("Negative Predictive Value (NPV)"), html.H2(f"{focus_npv:.1f}%", className='text-info')], className='text-center')
            ])
        ]),
        className='mb-4'
    )

def generate_results_summary(sens, spec):
    prevalence_points = [0.1, 1, 5, 10, 20, 50]
    ppv_vals, npv_vals = calculate_predictive_values(sens, spec, prevalence_points)
    lr_pos, lr_neg = calculate_likelihood_ratios(sens, spec)

    table_data = [{'Prevalence (%)': f'{p:.1f}', 'PPV (%)': f'{ppv:.1f}', 'NPV (%)': f'{npv:.1f}'} for p, ppv, npv in zip(prevalence_points, ppv_vals, npv_vals)]

    return dbc.Card(
        dbc.CardBody([
            html.H4([html.I(className="fas fa-table"), " Quick Reference Table"]),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                data=table_data,
                style_as_list_view=True,
                style_header={'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'padding': '10px'}
            ),
            html.Div([
                html.P([html.Strong("LR+ :"), f" {lr_pos:.2f}"], className='mt-3'),
                html.P([html.Strong("LR- :"), f" {lr_neg:.2f}"])
            ])
        ]),
        className='mb-4'
    )

def generate_clinical_insights(sens, spec, focus_prev):
    focus_ppv, focus_npv = calculate_predictive_values(sens, spec, focus_prev)
    lr_pos, lr_neg = calculate_likelihood_ratios(sens, spec)
    insights = []
    if focus_ppv < 50: insights.append(html.Li(f"At {focus_prev:.1f}% prevalence, the PPV is low ({focus_ppv:.1f}%). A positive result may often be a false positive.", className='text-warning'))
    else: insights.append(html.Li(f"At {focus_prev:.1f}% prevalence, the PPV is strong ({focus_ppv:.1f}%). A positive result is likely reliable.", className='text-success'))
    if focus_npv > 90: insights.append(html.Li(f"The NPV is excellent ({focus_npv:.1f}%). A negative result is highly reliable for ruling out disease.", className='text-success'))
    else: insights.append(html.Li(f"The NPV is moderate ({focus_npv:.1f}%). A negative result may still warrant caution.", className='text-warning'))
    if lr_pos >= 10: insights.append(html.Li(f"A high LR+ ({lr_pos:.2f}) strongly suggests disease when the test is positive.", className='text-success'))
    else: insights.append(html.Li(f"A low LR+ ({lr_pos:.2f}) means a positive test is not very informative.", className='text-danger'))
    if lr_neg <= 0.1: insights.append(html.Li(f"A low LR- ({lr_neg:.2f}) strongly rules out disease when the test is negative.", className='text-success'))
    else: insights.append(html.Li(f"A high LR- ({lr_neg:.2f}) means a negative test does not effectively rule out disease.", className='text-danger'))

    return dbc.Card(
        dbc.CardBody([
            html.H4([html.I(className="fas fa-lightbulb"), " Clinical Insights"]),
            html.Ul(insights)
        ])
    )

# --- Callbacks ---

@app.callback(Output('tab-content', 'children'), Input('mode-tabs', 'value'))
def render_tab_content(active_tab):
    tabs = {'calculator': create_calculator_tab, 'scenarios': create_scenarios_tab, 'comparison': create_comparison_tab, 'cases': create_cases_tab}
    return tabs.get(active_tab, lambda: html.Div("Select a tab"))()

@app.callback(
    [Output('main-container', 'className'), Output('theme-icon', 'className'), Output('theme-text', 'children'), Output('app-theme', 'data')],
    Input('theme-toggle', 'n_clicks'),
    State('app-theme', 'data'),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks and n_clicks % 2 == 1:
        return 'dark-theme', "fas fa-sun", "Light Mode", 'dark'
    return '', "fas fa-moon", "Dark Mode", 'light'


@app.callback(
    [Output('output-graph', 'children'), Output('results-summary', 'children'), Output('realtime-results', 'children'), Output('clinical-insights', 'children'),
     Output('sens', 'value', allow_duplicate=True), Output('spec', 'value', allow_duplicate=True), Output('focus-prevalence', 'value', allow_duplicate=True)],
    [Input('sens', 'value'), Input('spec', 'value'), Input('focus-prevalence', 'value'), Input('graph-type', 'value'),
     Input({'type': 'scenario-btn', 'index': dash.ALL}, 'n_clicks'), Input({'type': 'case-btn', 'index': dash.ALL}, 'n_clicks'), Input('app-theme', 'data')],
    prevent_initial_call='initial_duplicate'
def update_main_calculator(sens, spec, focus_prev, graph_type, scenario_clicks, case_clicks, theme):
    ctx = callback_context
    triggered_id = ctx.triggered_id

    if isinstance(triggered_id, dict):
        if triggered_id['type'] == 'scenario-btn':
            scenario_key = triggered_id['index']
            scenario = CLINICAL_SCENARIOS[scenario_key]
            sens, spec, focus_prev = scenario['sensitivity'], scenario['specificity'], scenario['typical_prevalence'][0]
        elif triggered_id['type'] == 'case-btn':
            case_key = triggered_id['index']
            case = CASE_STUDIES[case_key]
            sens, spec, focus_prev = case['test_params']['sensitivity'], case['test_params']['specificity'], case['prevalence']

    if sens is None or spec is None or focus_prev is None:
        return dash.no_update

    sens, spec, focus_prev = float(sens), float(spec), float(focus_prev)

    graph = create_main_graph(graph_type, sens, spec, focus_prev, theme)
    summary = generate_results_summary(sens, spec)
    realtime = generate_realtime_results(sens, spec, focus_prev)
    insights = generate_clinical_insights(sens, spec, focus_prev)

    return graph, summary, realtime, insights, sens, spec, focus_prev

@app.callback(
    [Output('comparison-tests', 'data'), Output('test-name', 'value'), Output('test-sens', 'value'), Output('test-spec', 'value')],
    Input('add-test-btn', 'n_clicks'),
    [State('comparison-tests', 'data'), State('test-name', 'value'), State('test-sens', 'value'), State('test-spec', 'value')],
    prevent_initial_call=True
)
def add_test_for_comparison(n_clicks, current_tests, name, sens, spec):
    if name and sens is not None and spec is not None:
        current_tests.append({'name': name, 'sensitivity': float(sens), 'specificity': float(spec)})
        return current_tests, '', None, None
    return current_tests, name, sens, spec

@app.callback(
    Output('comparison-results', 'children'),
    Input('comparison-tests', 'data'),
    State('app-theme', 'data')
)
def update_comparison_results(tests, theme):
    if not tests:
        return html.P("Add tests to compare their predictive values.", className='text-center')

    prev_range = np.linspace(0.1, 50, 200)
    fig = go.Figure()

    plot_bgcolor = '#2d2d2d' if theme == 'dark' else '#FFFFFF'
    paper_bgcolor = '#2d2d2d' if theme == 'dark' else '#FFFFFF'
    font_color = '#e0e0e0' if theme == 'dark' else '#2C3E50'
    grid_color = '#404040' if theme == 'dark' else '#E9ECEF'

    for i, test in enumerate(tests):
        ppv, npv = calculate_predictive_values(test['sensitivity'], test['specificity'], prev_range)
        color = go.layout.Template().layout.colorway[i % len(go.layout.Template().layout.colorway)]
        fig.add_trace(go.Scatter(x=prev_range, y=ppv, mode='lines', name=f"{test['name']} - PPV", line={'color': color, 'width': 3}))
        fig.add_trace(go.Scatter(x=prev_range, y=npv, mode='lines', name=f"{test['name']} - NPV", line={'color': color, 'width': 3, 'dash': 'dot'}))

    fig.update_layout(
        title_text="Predictive Value Comparison", title_x=0.5,
        xaxis_title="Disease Prevalence (%)", yaxis_title="Predictive Value (%)",
        hovermode='x unified', height=500,
        plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor,
        font={'color': font_color},
        xaxis={'gridcolor': grid_color}, yaxis={'gridcolor': grid_color},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})

@app.callback(
    Output("download-pdf-report", "data"),
    Input("export-btn", "n_clicks"),
    [State("sens", "value"), State("spec", "value"), State("focus-prevalence", "value")],
    prevent_initial_call=True
)
def generate_pdf_report(n_clicks, sens, spec, focus_prev):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Recalculate values for the report
    focus_ppv, focus_npv = calculate_predictive_values(sens, spec, focus_prev)
    lr_pos, lr_neg = calculate_likelihood_ratios(sens, spec)
    prevalence_points = [0.1, 1, 5, 10, 20, 50]
    ppv_vals, npv_vals = calculate_predictive_values(sens, spec, prevalence_points)

    elements = [
        Paragraph("Predictive Value Calculator Report", styles['h1']),
        Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"<b>Parameters:</b> Sensitivity: {sens}%, Specificity: {spec}%, Focus Prevalence: {focus_prev}%", styles['h2']),
        Spacer(1, 12),
        Paragraph(f"<b>Results at Focus Prevalence:</b> PPV: {focus_ppv:.1f}%, NPV: {focus_npv:.1f}%", styles['h2']),
        Spacer(1, 12),
        Paragraph("<b>Quick Reference Table:</b>", styles['h2'])
    ]

    table_data = [['Prevalence (%)', 'PPV (%)', 'NPV (%)']] + [[f'{p:.1f}', f'{ppv:.1f}', f'{npv:.1f}'] for p, ppv, npv in zip(prevalence_points, ppv_vals, npv_vals)]

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0,0), (-1,0), rl_colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), rl_colors.HexColor('#F8F9FA')),
        ('GRID', (0,0), (-1,-1), 1, rl_colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Likelihood Ratios:</b> LR+: {lr_pos:.2f}, LR-: {lr_neg:.2f}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<i>Graph image embedding is not supported in this version.</i>", styles['Italic']))

    doc.build(elements)
    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), "predictive_value_report.pdf")


if __name__ == '__main__':
    app.run_server(debug=True)
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div([
    html.H2("Positive and Negative Predictive Value Calculator"),
    html.P("Enter the sensitivity and specificity values for your test below. These values are usually published by the test manufacturer or a regulating body like the FDA."),
    html.Div([
        html.P(["Sensitivity: ",
              dcc.Input(id='sens', value=90, type='number', style={'width': 45}, min="0", max="100"),
              " = the percent of positive samples that are correctly identified by the test."]),
        html.P(["Specificity: ",
              dcc.Input(id='spec', value=85, type='number', style={'width': 45}, min="0", max="100"),
             " = the percent of negative samples that are correctly identified by the test."]),
    ], style={'marginLeft': 25}),
    html.P("The graph below shows how positive and negative predictive values vary with prevalence. See definitions below the graph."),
    html.Div(id='output-graph', style = {'max-width': '900px'}),
    html.H4("Definitions"),
    html.Div([
        html.P("Positive Predictive Vaule (PPV) = Percent of positive test results that are correct"),
        html.P("Negative Predictive Value (NPV) = Percent of negative test results that are correct"),
        html.P("Prevalence = The percent of a population who should test positive, e.g. the percent who have the disease you're testing for."),
    ], style={'marginLeft': 25}),
    html.H4("Example"),
    html.P(["A test with 90% sensitivity and 85% specificity used on a population with 5% prevalence would have a 24% PPV and 99% NPV.", html.Br(), "This means, on average, 24% of the positive test results would be correct and 99% of the negative test results would be correct."], style={"marginLeft": 25}),
    html.P([
        "For a more detailed yet laymen discussion of PPV and NPV, see ",
        html.A("the article I posted on Medium using COVID-19 antibody tests as an example.", href="https://medium.com/@liketortilla/can-you-trust-your-antibody-test-results-89e438b9bd4c", target="_blank"),
    ]),
    html.Br(),
    html.Pre([
        "Created by ",
        html.A("Derek Padilla", href="https://derekpadilla.com", target="_blank"),
        " using Plotly's ",
        html.A("Dash", href="https://plotly.com/dash/", target="_blank"),
        " Python framework.",
    ]),
    html.Br(),
])

@app.callback(
    Output('output-graph', 'children'),
    Input ('sens', 'value'),
    Input ('spec', 'value'))
def update_value(sens,spec):
    prev=np.linspace(0, 100, 1001)
    ppv=100*(sens*prev)/((sens*prev)+((100-spec)*(100-prev)))
    npv=100*(spec*(100-prev))/(((100-sens)*prev)+((spec*(100-prev))))

    return dcc.Graph(
        id='graph',
        figure={
            'data': [
                {'x': prev, 'y': ppv, 'type': 'line', 'name': 'PPV', 'line':dict(color='royalblue', width=6)},
                {'x': prev, 'y': npv, 'type': 'line', 'name': 'NPV', 'line':dict(color='firebrick', width=6)},
            ],
            'layout': {
                'legend': {
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': 1.02,
                    'xanchor': 'right',
                    'x': 0.2
                },
                'xaxis':{'title':'Prevalence'},
                'yaxis':{'title':'PPV / NPV'},
                'margin':{'t':0}
            }
        }
    )

if __name__ == '__main__':
    app.run_server()

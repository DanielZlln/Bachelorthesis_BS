import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor
import pickle
import csv
import json
import dash_table



with open('notebooks/best_random_forest_model_without_weather.pkl', 'rb') as rf_model_file:
    best_model_rf = pickle.load(rf_model_file)
    
with open('notebooks/model_metadata/features.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    features = next(reader)

df_pred = pd.read_csv('notebooks/model_metadata/predictions.csv')
df_tree = pd.read_csv('notebooks/model_metadata/df_tree.csv')
df_tree.drop(columns=['Neutor (gesamt)'], inplace=True)

results_df = pd.read_csv('notebooks/model_metadata/model_results.csv')
mae_random_forest = results_df.loc[results_df['Model'] == 'RANDOM FOREST', 'MAE'].values[0]
rmse_random_forest = results_df.loc[results_df['Model'] == 'RANDOM FOREST', 'RMSE'].values[0]
smape_random_forest = results_df.loc[results_df['Model'] == 'RANDOM FOREST', 'SMAPE'].values[0]

explainer_rf = LimeTabularExplainer(df_tree[features].values, 
                                     feature_names=features, 
                                     mode='regression')

# i = Zeitpunkt in dem DF
i = 150

# LIME-Erklärung
exp_rf = explainer_rf.explain_instance(df_tree[features].iloc[i].values, best_model_rf.predict)

lime_data = exp_rf.as_list()
lime_df = pd.DataFrame(lime_data, columns=['Feature', 'Weight'])


app = dash.Dash(__name__)
# The next column is important for AWS elastic beanstalk, without it wouldn't run
application = app.server 

app.layout = html.Div(
        children=[
        html.Div(
            className='first-row',
            children=[
                html.Img(
                    src=r'assets/logo-sw-rahmen.jpg',
                    alt='image',
                    style={
                        "height": "60px",
                        "width": "auto"
                    }
                )
            ]
        ),
        # Erste Reihe
        html.Div(
            className='first-row',
            children=[
                html.Div(
                    className='box small',
                    children=[
                        html.H6(children='MAE',
                                style={
                                    'textAlign': 'center',
                                    'color': 'white',
                                    'margin-top': '10px',
                                    'fontSize': 18
                                }),
                        html.P(f'{mae_random_forest:.2f}', 
                                style={
                                    'textAlign': 'center',
                                    'color': 'white',
                                    'fontSize': 30,
                                    'margin-top': '-40px'
                                })
                    ]
                ),
                html.Div(
                    className='box small',
                    children=[
                        html.H6(children='RMSE',
                                style={
                                    'textAlign': 'center',
                                    'color': 'white',
                                    'margin-top': '10px',
                                    'fontSize': 18
                                }),
                        html.P(f'{rmse_random_forest:.2f}',
                                style={
                                    'textAlign': 'center',
                                    'color': 'white',
                                    'fontSize': 30,
                                    'margin-top': '-40px'
                                })
                    ]
                ),
                html.Div(
                    className='box small',
                    children=[
                        html.H6(children='sMAPE',
                                style={
                                    'textAlign': 'center',
                                    'color': 'white',
                                    'margin-top': '10px',
                                    'fontSize': 18
                                }),
                        html.P(f'{smape_random_forest:.2f}',
                                style={
                                    'textAlign': 'center',
                                    'color': 'white',
                                    'fontSize': 30,
                                    'margin-top': '-40px'
                                })
                    ]
                )
            ]
        ),

        # Zweite Reihe
        html.Div(
            className='second-row',  
            children=[
                html.Div(className='box mid',
                        children=[
                            html.Div('Hier sind die tatsächlichen Werte der ersten Januar Woche 2023 und die Vorhersage von Random Forest')],
                        style={'text-align': 'center', 'font-size': '20px', 'color': 'white', 'line-height': '200%'}),
                html.Div(
                    className='box big',
                    style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'},
                    children=[
                        dcc.Graph(
                            figure=go.Figure(
                                data=[
                                    go.Line(
                                        name='Tatsächlich',
                                        x=df_pred['DatumZeit'],
                                        y=df_pred['y_original']
                                    ),
                                    go.Line(
                                        name='Random Forest',
                                        x=df_pred['DatumZeit'],
                                        y=df_pred['random_forest_predictions']
                                    )
                                ],
                                layout=go.Layout(
                                    title={
                                        'text': 'Tatsächlich vs. Random Forest Vorhersagen',
                                        'y': 0.9,
                                        'x': 0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},
                                    showlegend=True,
                                    xaxis=dict(title='Datum'),
                                    yaxis=dict(title='Wert'),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                            )
                        )
                    ])
                ]
        ),
        
        # Dritte Reihe
        html.Div(
            className='third-row',
            children=[
                html.Div(className='box mid',
                        children=[
                            html.Div('Hier sind die einzelnen lokalen Erklärungen von Lime für Random Forest. Hier wurde der selbe Tag ausgewählt wie in der Bachelorthesis')],
                        style={'text-align': 'center', 'font-size': '20px', 'color': 'white', 'line-height': '200%'}),
                html.Div(
                    className='box big',
                    style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'align-items': 'center'},
                    children=[
                        html.Div(
                            style={'text-align': 'center'},
                            children=[
                                html.H2("LIME Random Forest", style={'color': 'white'})
                            ]
                        ),
                        dash_table.DataTable(
                            id='lime-table',
                            columns=[
                                {'name': 'Feature', 'id': 'Feature'},
                                {'name': 'Weight', 'id': 'Weight'}
                            ],
                            data=lime_df.to_dict('records'),
                            style_table={
                                'maxHeight': '300px',
                                'overflowY': 'auto',
                                'border': 'none' 
                            },
                            style_cell={
                                'backgroundColor': 'rgba(0,0,0,0)',
                                'color': 'white',
                                'border': 'none'  
                            }
                        )
                    ]
                )
            ]
        )
    ]
)

if __name__ == '__main__':
    application.run(debug=True)

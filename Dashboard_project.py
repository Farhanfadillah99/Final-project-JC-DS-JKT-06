import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go
from dash.dependencies import Input, Output,State


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('retailMarketingDI.csv')
pred_model = pickle.load(open('project_modeling_1', 'rb'))


app.layout = html.Div(children=[
        html.H1('Dashboard'),
        html.P(html.H1('Project Retail Marketing')),
        dcc.Tabs(value = 'tabs',id='tabs-1',children=[
            dcc.Tab(label = 'Dataframe-Table',value='tab-1',children=[
                dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                   ),
           ]),

           dcc.Tab(label = 'Bar-Chart', value = 'tab-satu',children=[
                html.Div(children=[
                    html.Div(children=[html.H5('X1:'),
                        dcc.Dropdown(id = 'contoh-dropdown',
                        options = [{'label':'Salary','value':'Salary'},
                              {'label':'Children','value':'Children'},
                              {'label':'Catalogs','value':'Catalogs'},
                              {'label':'AmountSpent','value':'AmountSpent'},],
                        value = 'Salary')
                    ],className = 'col-3'),

                    html.Div(children=[html.H5('X2:'),
                        dcc.Dropdown(id = 'contoh-dropdown1',
                        options = [{'label':'Salary','value':'Salary'},
                              {'label':'Children','value':'Children'},
                              {'label':'Catalogs','value':'Catalogs'},
                              {'label':'AmountSpent','value':'AmountSpent'},],
                        value = 'AmountSpent')
                    ],className = 'col-3')
                    ],className = 'row'),
                    html.Div(children = dcc.Graph(
                        id = 'Graph-bar',
                        figure = {
                            'data': [
                                {'x':df['Married'],'y':df['AmountSpent'],'type':'bar','name' : 'Amount Spent'},
                                {'x':df['Married'],'y':df['Salary'],'type':'bar','name':'Salary'}
                                ],
                                'layout':{'title':'Bar Chart'}
                            },
                        ),
                    ),
            ]),

            dcc.Tab(label='Scatter-Chart', value='tab-2',children=[
                html.Div(children = dcc.Graph(
                    id = 'graph-scatter',
                    figure = {'data': [
                        go.Scatter(
                            x = df[df['Age']==i]['AmountSpent'],
                            y = df[df['Age']==i]['Salary'],
                            text = df[df['Age']==i]['Married'],
                            mode='markers',
                            name = 'Age {}'.format(i)
                            )for i in df['Age'].unique()
                        ],
                        'layout':go.Layout(
                            xaxis = {'title':'Amount Spent'},
                            yaxis = {'title':'Salary'},
                            hovermode='closest'
                        ),
                    },
                ),
            )]),

           dcc.Tab(label = 'Pie-Chart',value = 'tab-3',children =[
                html.Div(children = dcc.Graph(
                    id = 'graph-pie',
                    figure= {'data' : [
                        go.Pie(
                            labels = ['Catalogs {}'.format (i) for i in list(df['Catalogs'].unique())],
                            values = [df.groupby('Catalogs').mean()['AmountSpent'][i]for i in list(df['Catalogs'].unique())],
                            sort = False)
                        ],
                        'layout': {'title':'Mean Pie Chart'}
                    },
                ),
                ),
                ]),

    
           dcc.Tab(label='Prediction', value='tab-4', children=[
                html.Div(children=[
                    html.H5('Salary Prediction', className='mx-auto'),
                    html.Div(children=[
                        html.Div(children=[
                            html.P('Age: ', className='ml-2 mr-2'),
                            dcc.Dropdown(
                            id='Predict_Age',
                            options=[
                                {'label': 'Young', 'value': 0},
                                {'label': 'Middle', 'value': 1},
                                {'label': 'Old', 'value': 1}
                            ],
                            value=''
                            ),
                            
                            html.P('Gender', className='ml-2 mr-2'),
                            dcc.Dropdown(
                            id='Predict_Gender',
                            options=[
                                {'label': 'Female', 'value': 0},
                                {'label': 'Male', 'value': 1}
                            ],
                            value=''
                            ),
                            
                            html.P('Own Home: ',className='ml-2 mr-2'),
                            dcc.Dropdown(
                            id='Predict_OwnHome',
                            options=[
                                {'label': 'Own', 'value': 0},
                                {'label': 'Rent', 'value': 1}
                            ],
                            value=''
                            ),
                            
                            html.P('Married: ', className='ml-2 mr-2'),
                            dcc.Dropdown(
                            id='Predict_Married',
                            options=[
                                {'label': 'Single', 'value': 0},
                                {'label': 'Married', 'value': 1}
                            ],
                            value=''
                            ),   
                            
                            html.P('Location: ', className='ml-2 mr-2'),
                            dcc.Dropdown(
                                id='Predict_Location',
                                options=[
                                    {'label': 'Far', 'value': 0},
                                    {'label': 'Close', 'value': 1}
                            ],
                            value=''
                            ), 
                            html.P('Catalogs: ', className='ml-2 mr-2'),
                            # dcc.Input(id='Predict_Catalogs', type='number', value=110),
                            dcc.Dropdown(
                                id='Predict_Catalogs',
                                options=[
                                    {'label': i, 'value': i} for i in range(6,25)
                                    # {'label': 'Close', 'value': 1}
                            ],value=''
                            ),
                            html.P('Children: ', className='ml-2 mr-2'),
                            # dcc.Input(id='Predict_Children', type='number', value=110),
                            dcc.Dropdown(
                                id='Predict_Children',
                                options=[
                                    {'label': i, 'value': i} for i in range(0,5)
                                    # {'label': 'Close', 'value': 1}
                            ],value=''
                            ),
                            html.P('AmountSpent: ', className='ml-2 mr-2'),
                            # dcc.Input(id='Predict_AmountSpent', type='number', value=110),
                            dcc.Dropdown(
                                id='Predict_AmountSpent',
                                options=[
                                    {'label': i, 'value': i} for i in range(100,1000,100)
                                    # {'label': 'Close', 'value': 1}
                            ],value=''
                            ),
                            # html.Button(children='Predict', id='predict', className='ml-2 btn btn-secondary')
                        ]), 
                       

                    html.Div(children=[
                        dbc.Button("Predict", id="predict-button", className="mr-2"),
                        # html.Span(id="predict-output", style={"vertical-align": "middle"}),

                    html.Div(children=[], id='predict-output')

                    ]),
                    ]),
        ]),
        ]),
]),

])

@app.callback(
    Output(component_id = 'Graph-bar', component_property = 'figure'),
    [Input(component_id = 'contoh-dropdown', component_property = 'value'),
    Input(component_id = 'contoh-dropdown1', component_property = 'value')]
)

def create_graph_bar (x1,x2):
    figure = {
                    'data': [
                        {'x':df['Married'],'y':df[x1],'type':'bar','name' : x1},
                        {'x':df['Married'],'y':df[x2],'type':'bar','name':x2}
                    ],
                    'layout':{'title':'Bar Chart'}
                }
    return figure







@app.callback(
    Output(component_id='predict-output', component_property='children'),
    [Input(component_id='predict-button', component_property='n_clicks')],
    [State(component_id='Predict_Age', component_property='value'),
    State(component_id='Predict_Gender', component_property='value'),
    State(component_id='Predict_OwnHome', component_property='value'),
    State(component_id='Predict_Married', component_property='value'),
    State(component_id='Predict_Location', component_property='value'),
    State(component_id='Predict_Children', component_property='value'),
    State(component_id='Predict_Catalogs', component_property='value'),
    State(component_id='Predict_AmountSpent', component_property='value')]
)

def predict_legendary(n_clicks, Predict_Age, Predict_Gender, Predict_OwnHome, Predict_Married, Predict_Location, Predict_Children, Predict_Catalogs, Predict_AmountSpent):
    if n_clicks is None:
        children = 'Please fill all needed value!'

    if Predict_Age == 0:
        age_young = 1
        age_old = 0
        age_middle = 1
    elif Predict_Age == 1:
        age_young = 0
        age_old = 1
        age_middle = 1


    if Predict_Catalogs == 0 or Predict_AmountSpent == 0 or Predict_Age == '' or Predict_Gender =='' or Predict_OwnHome == '' or Predict_Married == '' or Predict_Location =='':
        children = 'Please fill all needed value!'
    else:
        X_test= np.array([Predict_Children, Predict_Catalogs, Predict_AmountSpent, age_old, age_young, Predict_Gender, Predict_OwnHome, Predict_Married, Predict_Location]).reshape(1,-1)
        predict_result = pred_model.predict(X_test)
        children = [html.Br(),html.H1('your salary is {} '.format(str(predict_result)[1:-1]))]
    return children

if __name__ == '__main__':
    app.run_server(debug=True)
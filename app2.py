#######
# First Milestone Project: Develop a Stock Ticker
# dashboard that either allows the user to enter
# a ticker symbol into an input box, or to select
# item(s) from a dropdown list, and uses pandas_datareader
# to look up and display stock data on a graph.
######

# EXPAND STOCK SYMBOL INPUT TO PERMIT MULTIPLE STOCK SELECTION
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output, State
import pandas_datareader.data as web # requires v0.6.0 or later
from datetime import datetime
import pandas as pd
# import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from plotly.graph_objs import Figure

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

tools.set_credentials_file(username='datar-ai', api_key='E4QJCIANLS8CVYnX7G0X')

fig: Figure = tools.make_subplots(rows=11, cols=3,
                          print_grid=False)

h = .02  # step size in the mesh


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


names = ["Input Data", "Nearest Neighbors", "Linear SVM",
         "RBF SVM", "Gaussian Process", "Decision Tree",
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

app = dash.Dash()
server = app.server

nsdq = pd.read_csv('./data/NASDAQcompanylist.csv')
nsdq.set_index('Symbol', inplace=True)
options = []
for tic in nsdq.index:
    options.append({'label':'{} {}'.format(tic,nsdq.loc[tic]['Name']), 'value':tic})

app.layout = html.Div([
    html.H1('Stock Ticker Dashboard'),
    html.Div([
        html.H4('NASDAQ DataTable'),
        dt.DataTable(
            rows=nsdq.to_dict('records'),

            # optional - sets the order of columns
            # columns=sorted(nsdq.columns),
            columns=nsdq.columns,

            row_selectable=True,
            filterable=True,
            sortable=True,
            selected_row_indices=[],
            id='datatable-NASDAQ'
        )
    ]),
    html.Div([
        html.H3('Select stock symbols:', style={'paddingRight':'30px'}),
        dcc.Dropdown(
            id='my_ticker_symbol',
            options=options,
            value=['NVDA'],
            multi=True
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'30%'}),
    html.Div([
        html.H3('Select start and end dates:'),
        dcc.DatePickerRange(
            id='my_date_picker',
            min_date_allowed=datetime(2015, 1, 1),
            max_date_allowed=datetime.today(),
            start_date=datetime(2018, 1, 1),
            end_date=datetime.today()
        )
    ], style={'display':'inline-block'}),
    html.Div([
        html.Button(
            id='submit-button',
            n_clicks=0,
            children='Submit',
            style={'fontSize':24, 'marginLeft':'30px'}
        ),
    ], style={'display':'inline-block'}),
    dcc.Graph(
        id='my_graph',
        figure={
            'data': [
                {'x': [1,2], 'y': [3,1]}
            ]
        },
        config={'displayModeBar': False}
    ),
    html.Div([
        html.H3('Select a Machine Learning model:', style={'paddingRight':'30px'}),
        dcc.Dropdown(
            id='ml_model',
            options=[
                {'label': 'Classifier comparison', 'value': 'Classifier comparison'},
                {'label': 'Nearest Neighbors', 'value': 'Nearest Neighbors'},
                {'label': 'Linear SVM', 'value': 'Linear SVM'},
                {'label': 'RBF SVM', 'value': 'RBF SVM'},
                {'label': 'Gaussian Process', 'value': 'Gaussian Process'},
                {'label': 'Decision Tree', 'value': 'Decision Tree'},
                {'label': 'Random Forest', 'value': 'Random Forest'},
                {'label': 'Neural Net', 'value': 'Neural Net'},
                {'label': 'AdaBoost', 'value': 'AdaBoost'},
                {'label': 'Naive Bayes', 'value': 'Naive Bayes'},
                {'label': 'QDA', 'value': 'QDA'},
            ],
            placeholder="Select a Machine Learning model",
            value='Classifier comparison'
        )],style={'display':'inline-block', 'verticalAlign':'top', 'width':'50%'}),
    html.Div([
        html.Button(
            id='ml-submit-button',
            n_clicks=0,
            children='Submit',
            style={'fontSize':24, 'marginLeft':'30px'}
        ),
    ], style={'display':'inline-block'}),
    dcc.Graph(
        id='graph-Classifier-comparison',
        figure={
            'data': [
                {'x': [1, 2], 'y': [3, 1]}
            ]
        },
        config={'displayModeBar': False}
    )
])


@app.callback(
    Output('graph-Classifier-comparison', 'figure'),
    [Input('ml-submit-button', 'n_clicks')],
    [State('ml_model', 'value')])
def update_ml_graph(n_clicks,value):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    np.random.seed(1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]

    i = 1
    j = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        # cm = plt.cm.RdBu
        # cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        # Plot the training points
        training_points = go.Scatter(x=X_train[:, 0], y=X_train[:, 1], showlegend=False,
                                     mode='markers', marker=dict(color='red'))
        # and testing points
        testing_points = go.Scatter(x=X_test[:, 0], y=X_test[:, 1], showlegend=False,
                                    mode='markers', marker=dict(color='blue'))

        fig.append_trace(training_points, 1, j)
        fig.append_trace(testing_points, 1, j)

        # iterate over classifiers
        i = 2
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)

            trace = go.Contour(y=xx[0], z=Z, x=xx[0],
                               line=dict(width=0),
                               contours=dict(coloring='heatmap'),
                               # colorscale=matplotlib_to_plotly(cm, 300),
                               opacity=0.7, showscale=False)

            # Plot also the training points

            training_points = go.Scatter(x=X_train[:, 0], y=X_train[:, 1], showlegend=False,
                                         mode='markers', marker=dict(color='red'))
            # and testing points

            testing_points1 = go.Scatter(x=X_test[:, 0], y=X_test[:, 1], showlegend=False,
                                         mode='markers', marker=dict(color='blue'))

            fig.append_trace(training_points, i, j)
            fig.append_trace(testing_points, i, j)
            fig.append_trace(trace, i, j)

            i = i + 1
        j += 1

    for i in map(str, range(1, 34)):
        x = 'xaxis' + i
        y = 'yaxis' + i
        fig['layout'][y].update(showticklabels=False, ticks='',
                                showgrid=False, zeroline=False)
        fig['layout'][x].update(showticklabels=False, ticks='',
                                showgrid=False, zeroline=False)
    k = 0

    for x in map(str, range(1, 32, 3)):
        y = 'yaxis' + x
        fig['layout'][y].update(title=names[k])
        k = k + 1

    fig['layout'].update(height=2000)
    return fig


@app.callback(
    Output('my_graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('my_ticker_symbol', 'value'),
    State('my_date_picker', 'start_date'),
    State('my_date_picker', 'end_date')])
def update_graph(n_clicks, stock_ticker, start_date, end_date):
    start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = datetime.strptime(end_date[:10], '%Y-%m-%d')
    traces = []
    for tic in stock_ticker:
        df = web.DataReader(tic,'iex',start,end)
        traces.append({'x':df.index, 'y': df.close, 'name':tic})
    fig2 = {
        'data': traces,
        'layout': {'title':', '.join(stock_ticker)+' Closing Prices'}
    }
    return fig2


if __name__ == '__main__':
    app.run_server()

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import time
from datetime import datetime
import backtrader as bt
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import numpy as np

external_stylesheets = [dbc.themes.BOOTSTRAP]


# The best feature about plotly dash is the ability to work with dataframes
def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    headers = list(df.iloc[0].name)
    row = []
    for x in range(1, 1509):
        row.append(list(df.iloc[x].name))
    df1 = pd.DataFrame(row, columns=headers)
    starting_val = df1['value'][1]
    ending_val = df1['value'][1507]
    df2 = df1.loc[:, ~df1.columns.duplicated()]
    x = df2['datetime']
    y = df2['open']
    buy = df2['buy']
    sell = df2['sell']
    fig = go.Figure(data=go.Scatter(x=x, y=y, name=f'{csv_file}'))

    fig.add_trace(go.Scatter(x=x, y=buy,
                             mode='markers', name='sell'))

    fig.add_trace(go.Scatter(x=x, y=sell,
                             mode='markers', name='buy'))
    return fig


# Calculates the starting value and ending value
def cash_value(csv_file):
    df = pd.read_csv(f'{csv_file}')
    headers = list(df.iloc[0].name)
    row = []
    for x in range(1, 1509):
        row.append(list(df.iloc[x].name))
    df1 = pd.DataFrame(row, columns=headers)
    starting_val = df1['value'][1]
    ending_val = df1['value'][1507]
    return[starting_val, ending_val]


# TODO save csv in the backend as a blob when using Django
# New test and Saves the file, although saving and deleting it is not the Ideal solution
def new_test(value):
    # Cerebro logic
    class SmaCross(bt.SignalStrategy):
        def __init__(self):
            sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
            crossover = bt.ind.CrossOver(sma1, sma2)
            self.signal_add(bt.SIGNAL_LONG, crossover)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)

    data0 = bt.feeds.YahooFinanceData(dataname=value, fromdate=datetime(2014, 1, 1),
                                      todate=datetime(2019, 12, 31))
    cerebro.addwriter(bt.WriterFile, out=f'{value}.csv', csv=True, rounding=2)
    cerebro.adddata(data0)

    cerebro.run()


# Generates 3 different tables
def generate_table(i, csv_file, max_rows=10):
    df = pd.read_csv(csv_file)
    headers = list(df.iloc[0].name)
    row = []
    for x in range(1, 1509):
        row.append(list(df.iloc[x].name))
    df1 = pd.DataFrame(row, columns=headers)
    df1.drop(['Id', 'len', 'volume', 'openinterest', 'adjclose', 'Trades - Net Profit/Loss', 'SmaCross', 'Broker', 'BuySell'], axis=1, inplace=True)
    # Moves the more important columns to the front
    buy_col = df1.pop('buy')
    sell_col = df1.pop('sell')
    value_col = df1.pop('value')
    df1.insert(0, 'buy', buy_col)
    df1.insert(0, 'sell', sell_col)
    df1.insert(0, 'value', value_col)

    starting_val = df1['value'][1]
    ending_val = df1['value'][1507]
    df2 = df1.loc[:, ~df1.columns.duplicated()]
    buydf = df2[pd.notnull(df2['buy'])]
    selldf = df2[pd.notnull(df2['sell'])]
    # Show different dataframes in accordion
    if i == 1:
        return dbc.Table.from_dataframe(df2.head(), responsive='sm', style={'overflowX': 'auto'},size='sm', striped=True, bordered=True, hover=True)
    elif i == 2:
        return dbc.Table.from_dataframe(buydf, responsive='sm', style={'overflowX': 'auto'},size='sm', striped=True, bordered=True, hover=True)
    elif i == 3:
        return dbc.Table.from_dataframe(selldf, responsive='sm', style={'overflowX': 'auto'},size='sm', striped=True, bordered=True, hover=True)


# This is our landing page for the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dbc.Jumbotron(
        [
            html.H1("Backtrader & Plotly Dash", className="display-3"),
            html.P(
                "Backtrader is a Algo trading Library that can Papertrade and "
                "Livetrade stocks by using strategies called 'signals'."
                "In the example below we input a stock symbol and use the "
                "long crossover strategy to trade $10000 over 5 years.",
                className="lead",
            ),
            dcc.Location(id='url', refresh=False),
            dcc.Link('Find out more about Backtrader', href='https://www.backtrader.com/'),
            html.Hr(className="my-2"),
            html.Div(dbc.Input(id='input-box', bs_size="lg", className="mb-3", type='text')),
            dbc.Button('Backtest', size="lg", outline=True, color="success", className="mr-1", id='loading-button'),
        ],
    ),
    dbc.Spinner(html.Div(id='output-container-button', children=' '),color="success")
])


def make_item(i, value):

    def return_string(i):
        if i == 1:
            return "Strategy head"
        elif i == 2:
            return "Strategy Buy Events"
        elif i == 3:
            return "Strategy Sell Events"

    # we use this function to make the example items to avoid code duplication
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H2(
                    dbc.Button(
                        return_string(i),
                        color="link",
                        id=f"group-{i}-toggle",
                    )
                )
            ),
            dbc.Collapse(
                dbc.CardBody(generate_table(i, f'{value}.csv')),
                id=f"collapse-{i}",
            ),
        ]
    )


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('loading-button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value='AMZN'):
    try:
        new_test(value)
        return html.Div([
            html.Div(dcc.Graph(figure=process_csv(f'{value}.csv'))),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4(f"Long Crossover Strategy for {value}"),
                        html.P(""),
                        html.H5("Starting Cash " + str(round(float(cash_value(f'{value}.csv')[0]), 2)),
                                className="card-subtitle"),
                        html.P(""),
                        html.H5("Ending Cash " + str(round(float(cash_value(f'{value}.csv')[1]), 2)),
                                className="card-subtitle"),
                        html.P(""),
                        html.P(
                            "In a long crossover strategy, the bot trades stocks according to the "
                            "moving average, with an intent of holding.",
                            className="card-text",
                        ),
                    ]
                ),
                style={"width": "50%", 'margin': '5rem'},
            ),
            html.Div(
                [make_item(1, value), make_item(2, value), make_item(3, value)], className="accordion"
            ),
        ])
    except:
        return dbc.Alert("Please enter a valid NASDAQ stock symbol (try AAPL)", color="success", style={"width": "50%", 'margin': '5rem'})


@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in range(1, 4)],
    [Input(f"group-{i}-toggle", "n_clicks") for i in range(1, 4)],
    [State(f"collapse-{i}", "is_open") for i in range(1, 4)],
)
def toggle_accordion(n1, n2, n3, is_open1, is_open2, is_open3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "group-1-toggle" and n1:
        return not is_open1, False, False
    elif button_id == "group-2-toggle" and n2:
        return False, not is_open2, False
    elif button_id == "group-3-toggle" and n3:
        return False, False, not is_open3
    return False, False, False


if __name__ == '__main__':
    app.run_server(debug=True)

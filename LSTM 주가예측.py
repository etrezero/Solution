import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from pykalman import KalmanFilter

# Dash Application
app = dash.Dash(__name__)

# Define the list of tickers
tickers = [
        'VUG', 'VTV', 'SOXX', 'SPY', 
        'QQQ', 'ACWI', 'NVDA', 'TSLA', 
        'IAUM', 'VWO', 'VEA'
]

app.layout = html.Div(children=[
    html.H1(children='COVENANT ETF/주가 Prediction', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Select Ticker Symbol:', style={'display': 'block', 'textAlign': 'center'}),
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': ticker, 'value': ticker} for ticker in tickers],
            value='VUG',  # Default value
            style={'width': '50%', 'margin': 'auto'}
        ),
        html.Button('Predict', id='predict-button', n_clicks=0),
    ], style={'marginBottom': '20px', 'textAlign': 'center'}),

    html.Div(id='expected-return', style={'textAlign': 'center'}),

    dcc.Graph(
        id='predicted-graph',
        style={
            'width': '75%',
            'margin': 'auto'
        }
    )
])

def apply_kalman_filter(prices, time_step):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, _ = kf.filter(prices.values[-time_step:])
    return state_means.flatten()

def predict_with_kalman_filter(kalman_filtered_data, time_step, num_days):
    predicted_prices = []
    for _ in range(num_days):
        next_price = kalman_filtered_data[-1]  # Use last filtered value as next prediction
        predicted_prices.append(next_price)
        kalman_filtered_data = np.append(kalman_filtered_data[1:], next_price)
    return np.array(predicted_prices)

@app.callback(
    [Output('predicted-graph', 'figure'), Output('expected-return', 'children')],
    [Input('predict-button', 'n_clicks')],
    [Input('ticker-dropdown', 'value')]
)
def update_graph(n_clicks, ticker):
    if n_clicks > 0:
        # Step 1: Load data for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        try:
            data = fdr.DataReader(ticker, start_date, end_date)
        except:
            return {}, f"Failed to load data for ticker symbol: {ticker}"

        # Step 2: Preprocess the data
        data['Close'] = data['Close'].astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']].values)

        time_step = 60  # We will use the last 60 days to predict the next day's close

        # Step 3: Apply Kalman Filter and predict the next 7 days
        kalman_filtered_data = apply_kalman_filter(data['Close'], time_step)
        kalman_predicted_prices = predict_with_kalman_filter(kalman_filtered_data, time_step, 7)

        # Step 4: Create the dataset for LSTM
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        X, y = create_dataset(scaled_data, time_step)

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Step 5: Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Step 6: Train the model
        model.fit(X, y, batch_size=1, epochs=1)

        # Step 7: Predict the next 7 days using LSTM
        last_60_days = scaled_data[-time_step:]
        X_test = last_60_days.reshape(1, -1, 1)
        lstm_predicted_prices = []
        for _ in range(7):
            predicted_price = model.predict(X_test)
            lstm_predicted_prices.append(predicted_price[0, 0])
            X_test = np.append(X_test[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

        # Step 8: Inverse transform to get the actual prices for LSTM
        lstm_predicted_prices = scaler.inverse_transform(np.array(lstm_predicted_prices).reshape(-1, 1))

        # Inverse transform for Kalman predictions (since they were not scaled)
        kalman_predicted_prices = np.array(kalman_predicted_prices).reshape(-1, 1)

        # Step 9: Calculate the average of LSTM and Kalman predictions
        average_predicted_prices = (lstm_predicted_prices + kalman_predicted_prices) / 2

        # Step 10: Calculate Bollinger Bands (1 sigma)
        std_dev = np.std(average_predicted_prices)
        upper_band = average_predicted_prices + std_dev
        lower_band = average_predicted_prices - std_dev

        # Calculate the expected percentage change for the next week based on LSTM
        initial_price = data['Close'].iloc[-1]
        predicted_price_next_week = average_predicted_prices[-1][0]
        expected_return = ((predicted_price_next_week - initial_price) / initial_price) * 100

        # Calculate returns for Bollinger Bands
        upper_band_return = ((upper_band[-1][0] - initial_price) / initial_price) * 100
        lower_band_return = ((lower_band[-1][0] - initial_price) / initial_price) * 100

        # Update the graph and return the expected return
        figure = {
            'data': [
                go.Scatter(
                    x=pd.date_range(end=end_date + timedelta(days=7), periods=7, freq='D'),
                    y=lstm_predicted_prices.flatten(),
                    mode='lines+markers',
                    name='LSTM Predicted Prices'
                ),
                go.Scatter(
                    x=pd.date_range(end=end_date + timedelta(days=7), periods=7, freq='D'),
                    y=kalman_predicted_prices.flatten(),
                    mode='lines+markers',
                    name='Kalman Filter Predicted Prices'
                ),
                go.Scatter(
                    x=pd.date_range(end=end_date + timedelta(days=7), periods=7, freq='D'),
                    y=upper_band.flatten(),
                    mode='lines',
                    name='Upper Bollinger Band (1σ)',
                    line=dict(dash='dash', color='rgba(255, 0, 0, 0.5)')
                ),
                go.Scatter(
                    x=pd.date_range(end=end_date + timedelta(days=7), periods=7, freq='D'),
                    y=lower_band.flatten(),
                    mode='lines',
                    name='Lower Bollinger Band (1σ)',
                    line=dict(dash='dash', color='rgba(0, 0, 255, 0.5)')
                ),
                go.Scatter(
                    x=data.index[-time_step:],
                    y=data['Close'].iloc[-time_step:],
                    mode='lines',
                    name='Actual Prices (Last 60 days)'
                )
            ],
            'layout': go.Layout(
                title=f'{ticker} Stock Price Prediction for Next 7 Days',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'},
                hovermode='x',
                annotations=[
                    dict(
                        x=0.5,
                        y=0.5,
                        xref='paper',
                        yref='paper',
                        text="Covenant",
                        showarrow=False,
                        font=dict(size=50, color='rgba(0, 0, 0, 0.4)'),
                        align='center',
                        opacity=0.4
                    ),
                    dict(
                        x=pd.date_range(end=end_date + timedelta(days=7), periods=7, freq='D')[-1],
                        y=upper_band[-1][0],
                        xref='x', yref='y',
                        text=f"Upper Band Return: {upper_band_return:.1f}%",
                        showarrow=True,
                        arrowhead=7,
                        ax=20,
                        ay=-40
                    ),
                    dict(
                        x=pd.date_range(end=end_date + timedelta(days=7), periods=7, freq='D')[-1],
                        y=lower_band[-1][0],
                        xref='x', yref='y',
                        text=f"Lower Band Return: {lower_band_return:.1f}%",
                        showarrow=True,
                        arrowhead=7,
                        ax=20,
                        ay=40
                    )
                ]
            )
        }

        return figure, f'1 Week Expected return: {expected_return:.2f}%'

    return {}, ""

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0")

from datetime import datetime
import config, csv
from binance.client import Client
import asyncio
from binance import AsyncClient, DepthCacheManager, BinanceSocketManager
from matplotlib import pyplot as plt
import pprint,json,numpy
from flask import jsonify
import pandas as pd
import datetime
from fbprophet import Prophet
from fbprophet.plot import plot, plot_plotly, plot_components_plotly


client = Client(config.API_KEY, config.API_SECRET)

# prices = client.get_all_tickers()

# for price in prices:
#     print(price)

#csvfile = open('2020_15minutes.csv', 'w', newline='') 
#candlestick_writer = csv.writer(csvfile, delimiter=',')

candlesticks = client.get_historical_klines("BTCUSDT", AsyncClient.KLINE_INTERVAL_1MINUTE, "10 day ago UTC")


processed_closed_candlesticks = []
processed_time_candlesticks = []

for data in candlesticks:
        candlestick = { 
            "time": data[0] /1000, 
            "open": data[1],
            "high": data[2], 
            "low": data[3], 
            "close": data[4]
        }
        processed_closed_candlesticks.append(candlestick["close"])
        processed_time_candlesticks.append(candlestick["time"])

print(processed_closed_candlesticks)
print(processed_time_candlesticks)
timestamps = []
for i in processed_time_candlesticks:
    timestamp = datetime.datetime.fromtimestamp(i)
    timestamps.append(timestamp)
timestamp_cleaned = []
for i in timestamps:
    timestamp_clean = i.strftime('%Y-%m-%d %H:%M:%S')
    timestamp_cleaned.append(timestamp_clean)
dataCom = list(zip(pd.DatetimeIndex(timestamp_cleaned),processed_closed_candlesticks))
df = pd.DataFrame(data=dataCom,columns=["ds","y"])

print( df.head()) 
#Training Model
m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(df)
#making predictions
future = m.make_future_dataframe(periods=30,freq='min',include_history=False)
forecast = m.predict(future)

print(len(forecast['yhat']))
predicted_ds_y = numpy.array(forecast['yhat'][0]) - numpy.array(forecast['yhat'][len(forecast['yhat'])-1])

if predicted_ds_y < 0 :
    predicted_direction = "down!!!!"
    print(predicted_direction)
else:
    predicted_direction ="up!!!!"
    print(predicted_direction)


predicted_values = []
predicted_values = numpy.array(forecast['yhat'][-5:])
predicted_dates  = []
predicted_dates  = forecast['ds']
print(forecast[['ds','yhat']])
print(predicted_values)
print(predicted_dates)
from matplotlib import pyplot as plt
fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
plt.show()
#validation
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
#m_validate = cross_validation(m,horizon='365 days',period='100 day',initial='200 days')
#print(m_validate.head())
#m_performance = performance_metrics(m_validate)
#print(m_performance.head())




#print(forecast.head())
#print(forecast.tail())


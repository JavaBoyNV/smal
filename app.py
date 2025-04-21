import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from flask import Flask, render_template, jsonify
import csv
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import json

app = Flask(__name__)

def get_nifty50_data():
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365)
    nifty50 = yf.download("^NSEI", start=start_date, end=end_date)['Close']
    nifty50 = nifty50.dropna()
    return nifty50

# Load CSV with stock data (Company Name, Symbol, Sector, Ratings)
def load_stock_data():
    with open('./static/nifty50list.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        stocks = [row for row in reader]
    return stocks

# Get stock data from Yahoo Finance
def get_stock_data(symbols):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365)
    stock_data = {}
    for symbol in symbols:
        stock_data[symbol] = yf.download(symbol, start=start_date, end=end_date)['Close']
    return stock_data

# Calculate PE Ratios for each stock
def get_pe_ratios(symbols):
    pe_ratios = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        try:
            pe_ratio = stock.info['trailingPE']
            pe_ratios[symbol] = pe_ratio
        except KeyError:
            pe_ratios[symbol] = None
    return pe_ratios

# Get stock with minimum PE ratio for each sector
def select_min_pe_stocks(stocks, pe_ratios):
    sector_stocks = {}
    for stock in stocks:
        sector = stock['Sector']
        symbol = stock['Symbol']
        if sector not in sector_stocks:
            sector_stocks[sector] = []
        sector_stocks[sector].append((symbol, pe_ratios.get(symbol)))
    
    min_pe_stocks = []
    for sector, stocks_in_sector in sector_stocks.items():
        valid_stocks = [(symbol, pe) for symbol, pe in stocks_in_sector if pe is not None]
        if valid_stocks:
            min_pe_stock = min(valid_stocks, key=lambda x: x[1])
            min_pe_stocks.append(min_pe_stock)
    return min_pe_stocks

def prepare_stock_data_for_optimization(stock_data, selected_symbols):
    # Align all the stock data to the same dates and ensure that missing values are handled
    aligned_data = [stock_data[symbol].reindex(stock_data[selected_symbols[0]].index) for symbol in selected_symbols]
    
    # Combine into a single DataFrame
    selected_stock_data_df = pd.concat(aligned_data, axis=1)
    selected_stock_data_df.columns = selected_symbols  # Label the columns with stock symbols
    return selected_stock_data_df

# Markowitz Portfolio Optimization
def markowitz_optimization(stocks,stock_data, state):
    mu = expected_returns.mean_historical_return(stock_data)
    S = risk_models.sample_cov(stock_data)
    ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 0.1))
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance()

    adjusted_weights = cleaned_weights.copy()
    if state in["Correction State", "Bearish"]:
        ratings = {stock['Symbol']: int(stock['Ratings']) for stock in stocks}
        weighted_ratings = {symbol: cleaned_weights[symbol] * ratings.get(symbol, 3) for symbol in adjusted_weights}
        total_weighted_ratings = sum(weighted_ratings.values())
        normalized_weights = {symbol: weight / total_weighted_ratings for symbol, weight in weighted_ratings.items()}
        adjusted_weights = normalized_weights

    new_expected_return = sum(adjusted_weights[symbol] * mu[symbol] for symbol in adjusted_weights)

    # Convert weights to JSON for Chart.js
    chart_data = {
        "labels": list(adjusted_weights.keys()),
        "datasets": [{
            "data": [round(w * 100, 2) for w in adjusted_weights.values()]           
        }]
    }

    with open("./static/chart_data.json", "w") as f:
        json.dump(chart_data, f)
    
    return round(new_expected_return*100,2)

def calculate_sector_growth(stocks, stock_data):
    sector_growth = {}

    for stock in stocks:
        sector = stock['Sector']
        symbol = stock['Symbol']
        
        if symbol in stock_data and not stock_data[symbol].empty:
            start_price = stock_data[symbol].iloc[0]
            end_price = stock_data[symbol].iloc[-1]
            growth = ((end_price - start_price) / start_price) * 100

            if sector in sector_growth:
                sector_growth[sector].append(growth)
            else:
                sector_growth[sector] = [growth]

    # Compute average growth per sector
    avg_sector_growth = {sector: np.mean(growths) for sector, growths in sector_growth.items()}

    # Get top 5 growing sectors
    top_5_sectors = sorted(avg_sector_growth.items(), key=lambda x: x[1], reverse=True)[:5]

    chart_data = {
        "labels": [sector for sector, _ in top_5_sectors],
        "datasets": [{
            "label": "Sector Growth (%)",
            "data": [round(growth, 2) for _, growth in top_5_sectors],
            "backgroundColor": ["#ff6384", "#36a2eb", "#ffce56", "#4bc0c0", "#9966ff"]
        }]
    }

    with open("./static/sector_growth.json", "w") as f:
        json.dump(chart_data, f)


@app.route('/')
def index():

    nifty_data = get_nifty50_data()
    latest_price = float(nifty_data.tail(5).mean())
    highest_price = float(nifty_data.max())
    diff_percent = ((highest_price - latest_price) / highest_price) * 100.0
    mavg = float(nifty_data.rolling(window=100).mean().iloc[-1])
    print("The 100 Days Moving Average is: ",mavg)

    if latest_price > mavg:
        state = "Bullish"
        color = 'rgba(0, 255, 0, 0.5)'
    elif latest_price <= mavg:
        if diff_percent <= 5.0:
            state = "Pullback State"
            color = 'rgba(255, 255, 0, 0.5)'
        elif diff_percent <= 10.0:
            state = "Correction State"
            color = 'rgba(255, 100, 0, 0.5)'
        else:
            state = "Bearish"
            color = 'rgba(255, 0, 0, 0.5)'

    stocks = load_stock_data()
    symbols = [stock['Symbol'] for stock in stocks]
    
    # Get stock data and PE ratios
    stock_data = get_stock_data(symbols)
    pe_ratios = get_pe_ratios(symbols)
    
    # Select minimum PE ratio stocks for each sector
    min_pe_stocks = select_min_pe_stocks(stocks, pe_ratios)
    selected_symbols = [stock[0] for stock in min_pe_stocks]
    
    # Prepare stock data for optimization
    selected_stock_data_df = prepare_stock_data_for_optimization(stock_data,selected_symbols)
    
    # Apply Markowitz Model
    expected_returns = markowitz_optimization(stocks,selected_stock_data_df, state)
    calculate_sector_growth(stocks, stock_data)
    
    return render_template('index.html',state=state, color=color,alert=(state=="Bearish"),expected_return=expected_returns)

@app.route('/data')
def data():
    nifty_data = get_nifty50_data()
    data_points = {
        "labels": list(nifty_data.index.strftime('%Y-%m-%d')),
        "prices": list(map(float, nifty_data.values))
    }
    return jsonify(data_points)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9159, debug=False, use_reloader=False)
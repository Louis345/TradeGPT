# stock_analysis.py

import yfinance as yf
import pandas as pd
import sys
import re
import os
import requests
from dotenv import load_dotenv
import json
import time

TICKER_CACHE_FILE = "ticker_cache.json"

def load_environment():
    """
    Loads environment variables from the .env file.
    """
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("Error: ALPHA_VANTAGE_API_KEY not found in environment variables.")
        sys.exit(1)
    return api_key

def load_ticker_cache():
    """
    Loads the ticker cache from a JSON file.
    """
    if os.path.exists(TICKER_CACHE_FILE):
        with open(TICKER_CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_ticker_cache(cache):
    """
    Saves the ticker cache to a JSON file.
    """
    with open(TICKER_CACHE_FILE, "w") as file:
        json.dump(cache, file, indent=4)

def fetch_ticker_symbol(name, api_key, cache, retries=3, backoff_factor=2):
    """
    Fetches the ticker symbol for a given company or sector name using Alpha Vantage's API.
    Utilizes a cache to minimize API calls.
    """
    if name in cache:
        return cache[name]
    
    try:
        # Alpha Vantage's SYMBOL_SEARCH endpoint
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={name}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if 'bestMatches' in data and len(data['bestMatches']) > 0:
            # Return the first best match symbol
            symbol = data['bestMatches'][0]['1. symbol']
            cache[name] = symbol
            # Respect rate limits: 5 requests per minute
            time.sleep(12)  # 60 seconds / 5 requests = 12 seconds per request
            return symbol
        else:
            print(f"No ticker symbol found for '{name}'.")
            cache[name] = None
            return None
    except Exception as e:
        if retries > 0:
            wait = backoff_factor ** (3 - retries)
            print(f"Error fetching ticker for '{name}': {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
            return fetch_ticker_symbol(name, api_key, cache, retries-1, backoff_factor)
        else:
            print(f"Failed to fetch ticker for '{name}' after multiple attempts.")
            cache[name] = None
            return None

def extract_recommendations(analysis_data):
    """
    Extract Buy, Hold, Sell recommendations from the analysis data.
    Assumes structured JSON format as per the updated langchain_analysis.py.
    """
    recommendations = {
        'Buy': [],
        'Hold': [],
        'Sell': []
    }

    for chunk in analysis_data:
        if "actions_to_consider" in chunk:
            actions = chunk["actions_to_consider"]
            for action, tickers in actions.items():
                if action in recommendations:
                    # Split tickers by comma and strip whitespace
                    for ticker in tickers:
                        ticker = ticker.strip().upper()
                        if ticker:
                            recommendations[action].append(ticker)
    return recommendations

def fetch_stock_data(ticker, period="1y"):
    """
    Fetches historical stock data for the given ticker and period.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            print(f"No data found for ticker: {ticker}")
            return None
        return hist
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_moving_averages(hist, windows=[50, 200]):
    """
    Calculates Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
    """
    for window in windows:
        hist[f"SMA_{window}"] = hist['Close'].rolling(window=window).mean()
        hist[f"EMA_{window}"] = hist['Close'].ewm(span=window, adjust=False).mean()
    return hist

def generate_final_recommendation(hist):
    """
    Generates Buy/Sell/Hold recommendations based on SMA crossover strategy.
    - Buy when SMA_50 crosses above SMA_200 (Golden Cross).
    - Sell when SMA_50 crosses below SMA_200 (Death Cross).
    - Hold otherwise.
    """
    recommendation = "Hold"
    if len(hist) < 200:
        # Not enough data to calculate SMA_200
        return recommendation
    
    # Get the last two SMA values to detect crossover
    sma_short_prev = hist['SMA_50'].iloc[-2]
    sma_short_last = hist['SMA_50'].iloc[-1]
    sma_long_prev = hist['SMA_200'].iloc[-2]
    sma_long_last = hist['SMA_200'].iloc[-1]
    
    if sma_short_prev <= sma_long_prev and sma_short_last > sma_long_last:
        recommendation = "Buy"
    elif sma_short_prev >= sma_long_prev and sma_short_last < sma_long_last:
        recommendation = "Sell"
    
    return recommendation

def save_recommendations(recommendations, filename="trade_recommendations.txt"):
    """
    Saves the recommendations to a text file.
    """
    try:
        with open(filename, "w") as file:
            for ticker, action in recommendations.items():
                file.write(f"{ticker}: {action}\n")
        print(f"Trade recommendations saved to {filename}")
    except Exception as e:
        print(f"Error saving recommendations: {e}")

def main():
    """
    Main function to orchestrate the stock analysis.
    """
    if len(sys.argv) < 2:
        print("Usage: python stock_analysis.py <input_analysis_json_file> [output_text_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "trade_recommendations.txt"
    if len(sys.argv) >=3:
        output_file = sys.argv[2]
    
    # Load environment variables
    api_key = load_environment()
    
    # Load ticker cache
    ticker_cache = load_ticker_cache()
    
    # Check if the analysis file exists
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Load recommendations from the analysis file
    print(f"Loading recommendations from '{input_file}'...")
    with open(input_file, "r") as file:
        analysis_data = json.load(file)
    
    initial_recommendations = extract_recommendations(analysis_data)
    if not any(initial_recommendations.values()):
        print("No valid recommendations found in the input file.")
        sys.exit(1)
    
    print(f"Initial recommendations extracted: {initial_recommendations}")
    
    # Initialize final recommendations dictionary
    final_recommendations = {}
    
    # Analyze each ticker
    for action, tickers in initial_recommendations.items():
        for ticker in tickers:
            # If ticker is already in a standard format (e.g., "AAPL"), skip fetching symbol
            if re.match(r'^[A-Z]{1,5}$', ticker):
                symbol = ticker
            else:
                # Attempt to fetch ticker symbol using company name
                print(f"\nFetching ticker symbol for '{ticker}'...")
                symbol = fetch_ticker_symbol(ticker, api_key, ticker_cache)
                if not symbol:
                    print(f"Could not map '{ticker}' to a ticker symbol. Skipping.")
                    continue
            
            print(f"Fetching data for {symbol}...")
            hist = fetch_stock_data(symbol)
            if hist is None:
                final_recommendations[symbol] = "Hold"  # Default action if data not found
                continue
            
            print(f"Calculating moving averages for {symbol}...")
            hist = calculate_moving_averages(hist)
            
            print(f"Generating final recommendation for {symbol} based on SMA crossover...")
            final_action = generate_final_recommendation(hist)
            final_recommendations[symbol] = final_action
            print(f"Final Recommendation for {symbol}: {final_action}")
    
    # Save ticker cache
    save_ticker_cache(ticker_cache)
    
    # Save all final recommendations to a text file
    save_recommendations(final_recommendations, output_file)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import colorlover
import plotly.graph_objects as go
import json
import os
from polygon import RESTClient
from datetime import datetime, timedelta
from functools import lru_cache
import concurrent.futures

pd.options.mode.chained_assignment = None

API_KEY = "YOUR POLYGON API KEY"
client = RESTClient(API_KEY)

PERIOD = 730
INTERVAL  = 'day'

equities = {
    "SPY" : "S&P 500",
    "QQQ": "NAS 100",
    "QQQE": "Equal Weight NQ",
    "RSPD": "Equal Weight SPY",
    "IVW": "Large Cap Growth",
    "VOOV": "Large Cap Value",
    "IJK": "Mid Cap Growth",
    "IJJ": "Mid Cap Value",
    "IWM": "Russell 2000",
    "IWN": "Russell 2000 Value",
    "IWO": "Russell 2000 Growth",
    "USO": "United States Oil Fund",
    "IBIT": "iShares Bitcoin Trust",
    "GLD": "Gold",
    "LQD": "Investment Grade Credit",
    "HYG": "High Yield Credit",
    "SHY": "1-3 year treasuries",
    "IEF": "7-10 year treasuries",
    "TLT": "20+ year treasuries",

    "MAGS": "Magnificent Seven ETF",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "AAPL": "Apple",
    "NVDA": "Nvdia",
    "META": "Meta",

    "MTUM": "Large Cap Momentum",
    "XMMO": "Mid Cap Momentum",
    "XSMO": "Small Caps Momentum",
    "QUAL": "Quality",
    "USMV": "Min Volatility",
    "FFTY": "Growth",

    "XLRE":"Real Estate",
    "XLU": "Utilities",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLC": "Communications",
    "XBI": "Biotech",
    "XLK": "Technology",

    "SLV": "IShares:Silver Trust",
    "DBA": "Invesco DB MS Agri",
    "URA": "Glbl X Uranium ETF",
    "XME": "SPDR S&P Metals&Mining",
    "XOP": "SPDR S&P Oil&Gas Exp",
    "XES": "SPDR S&P Oil&Gas E&S",
    "GDXJ": "VanEck:Jr Gold Miners",

    "TAN": "Invesco Solar",
    "PBW": "Invesco WldHill CE",

    "WGMI": "Valkyrie Bitcoin Miners",

    "SMH": "VanEck:Semiconductor",
    "IGV": "iShares:Expand Tch-Sftwr",
    "XTL": "SPDR S&P Telecom",
    "XSW": "SPDR S&P Sftwre & Svc",
    "WCLD": "WisdomTree:Cloud Cmptng",
    "CIBR": "FT II:Nsdq Cybersecurity",

    "ARKK": "ARKK",
    "IPO": "Renaissance IPO",

    "PBJ": "Invesco Food & Beverage",
    "XRT": "SPDR S&P Retail",
    "RTH": "VanEck Retail ETF",
    "IBUY": "Amplify Online Retail",
    "PEJ": "Invesco Leisure and Ent",
    "DRIV": "Glbl X Auto & Elct Vhcls",

    "KRE": "SPDR S&P Reg Banking ETF",
    "KBWB": "Big Bank ETF",
    "KIE": "SPDR S&P Insurance ETF",

    "SCHH": "REIT ETF",
    "IBB": "Biotech",
    "XHS": "Healthcare Services",
    "XHE": "Healthcare Equipment",

    "PKB": "Invesco Building & Cons",
    "ITB": "Home Construction",
    "XAR": "Aerospace & Defence",
    "XHB": "SPDR S&P Homebuilders",

    "EWJ": "iShares:MSCI Japan",
    "EWZ": "iShares:MSCI Brazil",
    "EWW": "iShares:MSCI Mexico",
    "FXI": "iShares:China Large Cp",
    "KWEB": "KraneShs:CSI China Intrt",
    "GXC": "SPDR S&P China",
}



# Cache for expensive calculations
@lru_cache(maxsize=32)
def discrete_background_color_bins(n_bins=5, df_min=0, df_max=100):
    """Generate style conditions for color bins, now more efficient with caching"""
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [((df_max - df_min) * i) + df_min for i in bounds]
    styleConditions = []
    
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        if i == len(bounds) - 1:
            max_bound += 1

        backgroundColor = colorlover.scales[str(n_bins)]["div"]["RdYlGn"][i - 1]

        styleConditions.append(
            {
                "condition": f"params.value >= {min_bound} && params.value < {max_bound}",
                "style": {"backgroundColor": backgroundColor, "color": "#000"},
            }
        )

    return styleConditions

def apply_styling(df, columns):
    """Apply styling to columns"""
    style_conditions = {}
    
    for column in columns:
        col_min = df[column].min()
        col_max = df[column].max()
        style_conditions[column] = discrete_background_color_bins(5, col_min, col_max)
    
    return style_conditions

def fetch_ticker_data(ticker, start_str, end_str, interval):
    """Worker function to fetch data for a single ticker"""
    try:
        aggs = client.get_aggs(ticker, 1, interval, start_str, end_str)
        
        # Pre-allocate the dictionary with the expected size
        data_list = []
        for bar in aggs:
            data_list.append({
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume,
                'Date': pd.to_datetime(bar.timestamp, unit='ms')
            })
        
        if not data_list:
            return ticker, None
            
        # Create DataFrame in one go
        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)
        return ticker, df
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return ticker, None

def get_stock_data(period=730, interval="day"):
    """Optimized function to fetch stock data using parallel processing"""
    # Calculate start and end dates
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=period))
    
    # Convert dates to string format required by Polygon
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    stock_data = {}
    tickers = list(equities.keys())
    
    # Use parallel processing to fetch data
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all ticker fetch tasks
        future_to_ticker = {
            executor.submit(fetch_ticker_data, ticker, start_str, end_str, interval): ticker 
            for ticker in tickers
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, df = future.result()
            if df is not None:
                stock_data[ticker] = df
    
    if stock_data:
        # Create multi-index DataFrame
        multi_df = pd.concat(stock_data.values(), axis=1, keys=stock_data.keys())
        multi_df.columns.names = ['Ticker', 'Price']
        return multi_df
    else:
        return pd.DataFrame()

def compute_relative_prices(stock_data):
    """Compute relative prices against SPY"""
    if "SPY" not in stock_data.columns.levels[0]:
        print("SPY data not found")
        return stock_data.copy()
        
    spy_prices = stock_data["SPY"]
    relative_prices = stock_data.div(spy_prices, axis=0)
    return relative_prices

def percent_rank(pd_series, value, precision):
    """Calculate percent rank more efficiently"""
    if len(pd_series) <= 1:
        return 0
    return np.round((pd_series < value).mean(), precision) * 100 

def last_close(ticker, stock_data):
    """Get last close price efficiently"""
    try:
        return stock_data[ticker]["Close"].iloc[-1]
    except (KeyError, IndexError):
        return np.nan

def compute_indicators(ticker, stock_data):
    """Compute technical indicators more efficiently"""
    try:
        df = stock_data[ticker].copy()
        close = df["Close"]
        last_price = close.iloc[-1]
        
        # Calculate all percentage changes at once
        pct_changes = pd.DataFrame({
            "Daily_Pct_Change": close.pct_change(1) * 100,
            "Weekly_Pct_Change": close.pct_change(5) * 100,
            "Monthly_Pct_Change": close.pct_change(21) * 100
        })
        
        # Calculate moving averages
        ma_10 = close.rolling(window=10).mean()
        ma_20 = close.rolling(window=20).mean()
        
        # 52-Week High and Low (more efficient)
        high_52w = close.rolling(window=252).max()
        low_52w = close.rolling(window=252).min()
        
        # Get latest values
        result = {
            "Daily_Pct_Change": pct_changes["Daily_Pct_Change"].iloc[-1],
            "Weekly_Pct_Change": pct_changes["Weekly_Pct_Change"].iloc[-1],
            "Monthly_Pct_Change": pct_changes["Monthly_Pct_Change"].iloc[-1],
            "10_DMA": ma_10.iloc[-1],
            "20_DMA": ma_20.iloc[-1],
            "52_Week_High": high_52w.iloc[-1],
            "52_Week_Low": low_52w.iloc[-1],
            "10_DMA_Pct": ((last_price / ma_10.iloc[-1]) - 1) * 100,
            "20_DMA_Pct": ((last_price / ma_20.iloc[-1]) - 1) * 100,
            "52_Week_High_Pct": ((last_price / high_52w.iloc[-1]) - 1) * 100,
            "52_Week_Low_Pct": ((last_price / low_52w.iloc[-1]) - 1) * 100
        }
        
        return pd.Series(result)
    except Exception as e:
        print(f"Error computing indicators for {ticker}: {e}")
        return pd.Series()

def create_volume_bar(ticker, stock_data):
    """Create volume bar chart more efficiently"""
    try:
        df = stock_data[ticker].tail(25)
        if df.empty:
            return go.Figure()
            
        df = df.reset_index()
        
        # Create figure more efficiently
        fig = go.Figure(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            marker_color='grey'
        ))
        
        fig.update_layout(
            showlegend=False,
            hovermode=False,
            yaxis_visible=False,
            yaxis_showticklabels=False,
            xaxis_visible=False,
            xaxis_showticklabels=False,
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_dark",
            xaxis={'type':'category'}
        )
        
        return fig
    except Exception as e:
        print(f"Error creating volume bar for {ticker}: {e}")
        return go.Figure()

def create_rs_bar(ticker, stock_data):
    """Create RS bar chart more efficiently"""
    try:
        df = stock_data[ticker].tail(25)
        if df.empty:
            return go.Figure()
            
        df = df.reset_index()
        
        # Calculate scaled values more efficiently
        closes = df['Close'].values
        min_close = closes.min()
        max_close = closes.max()
        
        if max_close == min_close:  # Avoid division by zero
            scaled = np.ones(len(closes))
        else:
            # Log transform and scale in one step
            log_vals = np.log1p((closes - min_close) * 100)
            log_min = log_vals.min()
            log_max = log_vals.max()
            scaled = ((log_vals - log_min) / (log_max - log_min)) + 1
        
        # Set colors efficiently
        max_idx = np.argmax(scaled)
        min_idx = np.argmin(scaled)
        colors = ['green'] * len(scaled)
        colors[max_idx] = 'blue'
        colors[min_idx] = 'red'
        
        # Create traces for the chart
        fig = go.Figure()
        
        for i in range(len(df)):
            fig.add_trace(go.Bar(
                x=[df['Date'][i]], 
                y=[scaled[i]], 
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            showlegend=False,
            hovermode=False,
            yaxis_visible=False,
            yaxis_showticklabels=False,
            xaxis_visible=False,
            xaxis_showticklabels=False,
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_dark",
            xaxis={'type':'category'}
        )
        
        return fig
    except Exception as e:
        print(f"Error creating RS bar for {ticker}: {e}")
        return go.Figure()

@lru_cache(maxsize=1)
def read_comments_for_tickers(file_path):
    """Cache comments data to avoid repeated file reads"""
    # Check if file exists
    if not os.path.exists(file_path):
        return {}  # Return empty dict

    # Load JSON data
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            if not isinstance(data, list):  # Ensure it's a list
                return {}
    except json.JSONDecodeError:
        return {}  # Return empty dict if file is corrupt

    # Create a dictionary mapping tickers to their latest values
    ticker_map = {}
    for entry in data:
        for key, value in entry.items():
            ticker_map[key] = value  # Keep the latest occurrence

    return ticker_map

def get_comments(file_path, tickers):
    """Get comments for tickers using cached data"""
    ticker_map = read_comments_for_tickers(file_path)
    return [ticker_map.get(ticker, "") for ticker in tickers]

def generate_table():
    """Generate table data"""
    # Fetch stock data
    stock_data = get_stock_data(PERIOD, INTERVAL)
    
    # Early return if no data
    if stock_data.empty:
        return [], [], {}
    
    # Compute relative prices
    relative_prices_df = compute_relative_prices(stock_data)
    
    # Get tickers list once
    tickers = list(equities.keys())
    
    # Calculate all indicators in parallel
    indicators = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_ticker = {
            executor.submit(compute_indicators, ticker, stock_data): ticker 
            for ticker in tickers if ticker in stock_data.columns.levels[0]
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                indicators[ticker] = future.result()
            except Exception as e:
                print(f"Error with {ticker}: {e}")
                continue
    
    # Create visualization objects in parallel
    rs_graphs = {}
    volume_graphs = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit RS graph creation tasks
        rs_futures = {
            executor.submit(create_rs_bar, ticker, relative_prices_df): ticker 
            for ticker in tickers if ticker in relative_prices_df.columns.levels[0]
        }
        
        # Submit Volume graph creation tasks
        volume_futures = {
            executor.submit(create_volume_bar, ticker, stock_data): ticker 
            for ticker in tickers if ticker in stock_data.columns.levels[0]
        }
        
        # Collect RS results
        for future in concurrent.futures.as_completed(rs_futures):
            ticker = rs_futures[future]
            try:
                rs_graphs[ticker] = future.result()
            except Exception as e:
                print(f"Error creating RS graph for {ticker}: {e}")
                rs_graphs[ticker] = go.Figure()
        
        # Collect Volume results
        for future in concurrent.futures.as_completed(volume_futures):
            ticker = volume_futures[future]
            try:
                volume_graphs[ticker] = future.result()
            except Exception as e:
                print(f"Error creating volume graph for {ticker}: {e}")
                volume_graphs[ticker] = go.Figure()
    
    # Calculate RS percent ranks efficiently
    rs_values = {}
    for ticker in tickers:
        if ticker in relative_prices_df.columns.levels[0]:
            try:
                series = relative_prices_df[ticker]['Close'].tail(25)
                last_val = series.iloc[-1]
                rs_values[ticker] = percent_rank(series, last_val, 6)
            except Exception:
                rs_values[ticker] = np.nan
        else:
            rs_values[ticker] = np.nan
    
    # Create data dictionary
    data = {
        "comments": get_comments("./comments.json", tickers),
        "ticker": tickers,
        "company": [equities[ticker] for ticker in tickers],
        "price": [last_close(ticker, stock_data) if ticker in stock_data.columns.levels[0] else np.nan for ticker in tickers],
        "graph_rs": [rs_graphs.get(ticker, go.Figure()) for ticker in tickers],
        "rs": [rs_values.get(ticker, np.nan) for ticker in tickers],
    }
    
    # Add indicator data
    indicator_fields = [
        "daily_pct_change", "weekly_pct_change", "monthly_pct_change",
        "10_DMA", "20_DMA", "52_week_high", "52_week_low",
        "10_DMA_pct", "20_DMA_pct", "52_week_high_pct", "52_week_low_pct",
    ]
    
    indicator_map = {
        "daily_pct_change": "Daily_Pct_Change",
        "weekly_pct_change": "Weekly_Pct_Change",
        "monthly_pct_change": "Monthly_Pct_Change",
        "10_DMA": "10_DMA",
        "20_DMA": "20_DMA",
        "52_week_high": "52_Week_High",
        "52_week_low": "52_Week_Low",
        "10_DMA_pct": "10_DMA_Pct",
        "20_DMA_pct": "20_DMA_Pct",
        "52_week_high_pct": "52_Week_High_Pct",
        "52_week_low_pct": "52_Week_Low_Pct",
    }
    
    for field in indicator_fields:
        data[field] = [
            indicators.get(ticker, {}).get(indicator_map[field], np.nan) 
            for ticker in tickers
        ]
    
    # Add volume graph data
    data["graph_v"] = [volume_graphs.get(ticker, go.Figure()) for ticker in tickers]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate style conditions
    style_columns = [
        "rs", "daily_pct_change", "weekly_pct_change", "monthly_pct_change",
        "10_DMA_pct", "20_DMA_pct", "52_week_high_pct", "52_week_low_pct"
    ]
    
    style_conditions = apply_styling(df, style_columns)
    
    # Define column definitions
    columnDefs = [
        {
            "headerName": "Comments",
            "field": "comments",
            "editable": True
        },
        {
            "headerName": "Stock Ticker",
            "field": "ticker",
            "filter": True,
        },
        {
            "headerName": "Company",
            "field": "company",
            "filter": True,
        },
        {
            "headerName": "Price",
            "field": "price",
            "type": "rightAligned",
            "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"}
        },
        {
            "headerName": "RS",
            "field": "graph_rs",
            "cellRenderer": "DCC_Graph1",
            "maxWidth": 500,
            "minWidth": 200,
        },
        {
            "headerName": "RS%",
            "field": "rs",
            "type": "rightAligned",
            "cellStyle": {"styleConditions": style_conditions["rs"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        },
        {
            "field": "graph_v",
            "cellRenderer": "DCC_Graph2",
            "headerName": "Volume",
            "maxWidth": 500,
            "minWidth": 200,
        },
        {
            "headerName": "% Daily",
            "type": "rightAligned",
            "field": "daily_pct_change",
            "cellStyle": {"styleConditions": style_conditions["daily_pct_change"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        },
        {
            "headerName": "% Weekly",
            "type": "rightAligned",
            "field": "weekly_pct_change",
            "cellStyle": {"styleConditions": style_conditions["weekly_pct_change"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        },
        {
            "headerName": "% Monthly",
            "type": "rightAligned",
            "field": "monthly_pct_change",
            "cellStyle": {"styleConditions": style_conditions["monthly_pct_change"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        },
        {
            "headerName": "% 10 DMA",
            "type": "rightAligned",
            "field": "10_DMA_pct",
            "cellStyle": {"styleConditions": style_conditions["10_DMA_pct"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        },
        {
            "headerName": "% 20 DMA",
            "type": "rightAligned",
            "field": "20_DMA_pct",
            "cellStyle": {"styleConditions": style_conditions["20_DMA_pct"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        },
        {
            "headerName": "%52w Low",
            "type": "rightAligned",
            "field": "52_week_low_pct",
            "cellStyle": {"styleConditions": style_conditions["52_week_low_pct"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        },
        {
            "headerName": "%52w High",
            "type": "rightAligned",
            "field": "52_week_high_pct",
            "cellStyle": {"styleConditions": style_conditions["52_week_high_pct"]},
            "valueFormatter": {"function": "d3.format(',.2f')(params.value) + '%'"}
        }
    ]

    defaultColDef = {
        "resizable": True,
        "sortable": True,
        "editable": False,
    }

    return df.to_dict("records"), columnDefs, defaultColDef


def write_comments_to_json(file_path, new_entry):
    """Write comments to JSON file"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Read existing data if the file exists, otherwise start with an empty list
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            try:
                data = json.load(json_file)
                if not isinstance(data, list):  # Ensure it's a list
                    data = []
            except json.JSONDecodeError:
                data = []  # Handle case where file is empty or corrupted
    else:
        data = []

    # Extract key and value from new_entry (assumes a single key-value pair per dict)
    new_key, new_value = next(iter(new_entry.items()))

    # Check if the key exists in any dictionary
    updated = False
    for entry in data:
        if new_key in entry:
            entry[new_key] = new_value  # Update the value
            updated = True
            break

    # If no match was found, append the new key-value pair
    if not updated:
        data.append(new_entry)

    # Write the updated data back to the file
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    
    # Invalidate the cache to ensure fresh data on next read
    read_comments_for_tickers.cache_clear()
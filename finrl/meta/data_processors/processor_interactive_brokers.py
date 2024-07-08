from ib_insync import *
import pandas as pd
import pytz
import asyncio
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
from datetime import datetime
from stockstats import StockDataFrame as Sdf
import numpy as np

nest_asyncio.apply()

class IBProcessor:
    def __init__(self, IB_GATEWAY_HOST, IB_GATEWAY_PORT, CLIENT_ID):
        self.ib = IB()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.ib.connectAsync(IB_GATEWAY_HOST, IB_GATEWAY_PORT, clientId=CLIENT_ID))

    def format_ib_date(self, date_str, time_str='16:00:00', timezone='US/Eastern'):
        dt = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
        formatted_date = dt.strftime(f"%Y%m%d %H:%M:%S {timezone}")
        return formatted_date

    async def _fetch_data_for_ticker(self, symbol, start_date, end_date, duration, bar_size):
        contract = Stock(symbol, 'SMART', 'USD')
        end_date_formatted = self.format_ib_date(end_date)
        bars = None
        retries = 3
        for attempt in range(retries):
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract, endDateTime=end_date_formatted, durationStr=duration,
                    barSizeSetting=bar_size, whatToShow='TRADES', useRTH=True
                )
                break
            except ConnectionError as e:
                print(f"Connection error for {symbol} (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to fetch data for {symbol} after {retries} attempts")
        
        if bars:
            df = util.df(bars)
            df['symbol'] = symbol
            return df
        else:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()

    async def download_data(self, ticker_list, start_date, end_date, time_interval):
        self.start = start_date
        self.end = end_date
        
        # Convert time_interval to appropriate barSizeSetting
        if time_interval == '1D':
            bar_size = '1 day'
        elif time_interval == '1H':
            bar_size = '1 hour'
        else:
            raise ValueError(f"Unsupported time interval: {time_interval}")
        
        duration = '1 Y'  # You might want to calculate this based on start and end dates

        tasks = [
            self._fetch_data_for_ticker(ticker, start_date, end_date, duration, bar_size)
            for ticker in ticker_list
        ]
        data_list = await asyncio.gather(*tasks)

        data_df = pd.concat(data_list, axis=0)
        data_df['date'] = pd.to_datetime(data_df['date']).dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward')
        data_df = data_df.reset_index().rename(columns={'date': 'timestamp', 'symbol': 'tic'})
        data_df = data_df.sort_values(by=['tic', 'timestamp']).reset_index(drop=True)
        return data_df

    def clean_data(self, df):
        print("Data cleaning started")
        tic_list = df['tic'].unique()
        times = pd.date_range(self.start, self.end, freq='D').tz_localize('America/New_York')
        
        future_results = []
        for tic in tic_list:
            tmp_df = pd.DataFrame(index=times)
            tic_df = df[df['tic'] == tic].set_index('timestamp')
            tmp_df = tmp_df.join(tic_df, how='left').ffill().fillna(0)
            tmp_df['tic'] = tic
            future_results.append(tmp_df.reset_index().rename(columns={'index': 'timestamp'}))
        
        new_df = pd.concat(future_results).reset_index(drop=True)
        print("Data cleaning finished")
        return new_df

    def add_technical_indicator(self, df, tech_indicator_list):
        print("Adding technical indicators")
        stock_df = Sdf.retype(df.copy())
        for indicator in tech_indicator_list:
            df[indicator] = stock_df[indicator]
        print("Finished adding indicators")
        return df

    def add_vix(self, df):
        print("Adding VIX data")
        vix_contract = Future(symbol="VIX", lastTradeDateOrContractMonth='202309', exchange="CFE", currency="USD")
        
        try:
            vix_bars = self.ib.reqHistoricalData(
                vix_contract, endDateTime='', durationStr='1 Y', barSizeSetting='1 day',
                whatToShow='TRADES', useRTH=True
            )
            if vix_bars:
                vix_df = util.df(vix_bars).rename(columns={'date': 'timestamp', 'close': 'VIX'})
                df = df.merge(vix_df[['timestamp', 'VIX']], on='timestamp', how='left')
                df['VIX'] = df['VIX'].ffill()  # Forward fill VIX values
                print("Finished adding VIX data")
            else:
                print("No VIX data retrieved")
        except Exception as e:
            print(f"Error retrieving VIX data: {e}")
            print("Continuing without VIX data")
        
        return df

    def calculate_turbulence(self, df, window=252):
        print("Calculating turbulence")
        df = df.pivot(index='timestamp', columns='tic', values='close').pct_change()
        turbulence_index = df.rolling(window=window).apply(lambda x: np.linalg.det(x.cov()), raw=False)
        turbulence_index = turbulence_index.reset_index().melt(id_vars='timestamp', var_name='tic', value_name='turbulence')
        df = df.reset_index().melt(id_vars='timestamp', var_name='tic', value_name='close').merge(turbulence_index, on=['timestamp', 'tic'], how='left')
        print("Finished calculating turbulence")
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix=False):
        print("Converting DataFrame to arrays")
        print("DataFrame columns:", df.columns)
        print("DataFrame head:")
        print(df.head())
        
        # Ensure 'timestamp' is datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by timestamp and tic to remove duplicates
        df = df.groupby(['timestamp', 'tic']).agg({
            'close': 'last',
            'macd': 'last',
            'rsi_30': 'last'
        }).reset_index()
        
        # Check for remaining duplicates
        duplicates = df.duplicated(subset=['timestamp', 'tic'])
        if duplicates.any():
            print(f"Warning: {duplicates.sum()} duplicate entries found after grouping.")
            print("Duplicate entries:")
            print(df[duplicates])
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp', 'tic'])
        
        print("Pivoting DataFrame")
        price_array = df.pivot(index='timestamp', columns='tic', values='close').values
        tech_array = df.pivot(index='timestamp', columns='tic', values=tech_indicator_list).values
        
        # Handle NaN and inf values
        tech_array = np.nan_to_num(tech_array, nan=0, posinf=0, neginf=0)
        
        print("Price array shape:", price_array.shape)
        print("Tech array shape:", tech_array.shape)
        
        return price_array, tech_array

async def main():
    ib_processor = IBProcessor(IB_GATEWAY_HOST='127.0.0.1', IB_GATEWAY_PORT=7497, CLIENT_ID=1)
    data = await ib_processor.download_data(ticker_list=['AAPL', 'GOOG'], start_date='2023-01-01', end_date='2023-12-31', time_interval='1D')
    print("Raw data columns:", data.columns)
    print("Raw data head:")
    print(data.head())
    
    cleaned_data = ib_processor.clean_data(data)
    print("Cleaned data columns:", cleaned_data.columns)
    print("Cleaned data head:")
    print(cleaned_data.head())
    
    data_with_indicators = ib_processor.add_technical_indicator(cleaned_data, tech_indicator_list=['macd', 'rsi_30'])
    print("Data with indicators columns:", data_with_indicators.columns)
    print("Data with indicators head:")
    print(data_with_indicators.head())
    
    data_with_vix = ib_processor.add_vix(data_with_indicators)
    
    print("Final DataFrame columns:", data_with_vix.columns)
    print("Final DataFrame head:")
    print(data_with_vix.head())
    
    price_array, tech_array = ib_processor.df_to_array(data_with_vix, tech_indicator_list=['macd', 'rsi_30'])
    print("Price array shape:", price_array.shape)
    print("Tech array shape:", tech_array.shape)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if str(e).startswith("This event loop is already running"):
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
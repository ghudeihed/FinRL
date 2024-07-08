import asyncio
import json
import logging
import pandas as pd
import pytz
from datetime import datetime, timedelta
from stockstats import StockDataFrame as Sdf
import numpy as np
from ib_insync import *
import nest_asyncio

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IBProcessor:
    def __init__(self, IB_GATEWAY_HOST, IB_GATEWAY_PORT, start_client_id=1, client_id_file='client_id.json'):
        self.IB_GATEWAY_HOST = IB_GATEWAY_HOST
        self.IB_GATEWAY_PORT = IB_GATEWAY_PORT
        self.client_id_file = client_id_file
        self.ib = IB()
        self.current_client_id = self._load_client_id(start_client_id)

    def _load_client_id(self, default_id):
        try:
            with open(self.client_id_file, 'r') as f:
                data = json.load(f)
                return data.get('last_client_id', default_id) + 1
        except FileNotFoundError:
            return default_id

    def _save_client_id(self):
        with open(self.client_id_file, 'w') as f:
            json.dump({'last_client_id': self.current_client_id}, f)

    def connect(self):
        while True:
            try:
                logging.info(f"Attempting connection with client ID: {self.current_client_id}")
                self.ib.connect(self.IB_GATEWAY_HOST, self.IB_GATEWAY_PORT, clientId=self.current_client_id)
                if self.ib.isConnected():
                    logging.info(f"Connected with client ID: {self.current_client_id}")
                    self._save_client_id()
                    return
            except ConnectionError as e:
                logging.error(f"Connection error: {e}")
                self.current_client_id += 1  # Increment client ID only on ConnectionError
                if self.current_client_id > 100:  # Arbitrary upper limit to avoid infinite loop
                    logging.error("Exceeded maximum client ID retries")
                    raise

    def format_ib_date(self, date_str, time_str='16:00:00', timezone='US/Eastern'):
        dt = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
        formatted_date = dt.astimezone(pytz.timezone(timezone)).strftime(f"%Y%m%d %H:%M:%S {timezone}")
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
            except (ConnectionError, asyncio.TimeoutError) as e:
                logging.error(f"Error fetching data for {symbol} (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt) 
                else:
                    raise  # Re-raise the exception after retries

        if bars:
            df = util.df(bars)
            df['symbol'] = symbol
            return df
        else:
            logging.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

    async def download_data(self, ticker_list, start_date, end_date, time_interval):
        self.start = start_date
        self.end = end_date
        
        # Convert time_interval to appropriate barSizeSetting
        if time_interval == '1D':
            bar_size = '1 day'
            duration = f"{max(0, (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1)} D"
        elif time_interval == '1H':
            bar_size = '1 hour'
            duration = f"{max(0, (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days * 24 + 1)} H"  # Assuming 24 trading hours
        else:
            raise ValueError(f"Unsupported time interval: {time_interval}")

        tasks = [self._fetch_data_for_ticker(ticker, start_date, end_date, duration, bar_size) for ticker in ticker_list]
        data_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        data_list = [result for result in data_list if not isinstance(result, Exception)]
        for result in data_list:
            if isinstance(result, Exception):
                logging.error(f"Error downloading data: {result}")

        if not data_list:
            raise ValueError("No data fetched for any ticker")

        data_df = pd.concat(data_list, axis=0)
        data_df['date'] = pd.to_datetime(data_df['date']).dt.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
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
        """
        Calculate technical indicators using the stockstats package.

        :param df: (pd.DataFrame) Pandas dataframe containing financial data.
        :param tech_indicator_list: (list) List of technical indicators to calculate.
        :return: (pd.DataFrame) Pandas dataframe with added technical indicators.
        """
        print(f"Adding technical indicators {tech_indicator_list}...")
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for tic in unique_ticker:
                try:
                    temp_indicator = stock[stock.tic == tic][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = tic
                    # Rename the index column to 'timestamp' within temp_indicator
                    temp_indicator = temp_indicator.rename_axis('timestamp').reset_index() 
                    indicator_df = pd.concat([indicator_df, temp_indicator], axis=0, ignore_index=True)
                except Exception as e:
                    print(e)
            # Rename 'index' to 'timestamp' after concatenation for consistency
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]], on=["tic", "timestamp"], how="left"
            )

            print(f"Finished adding indicators: {indicator}")

        df = df.sort_values(by=["tic", "timestamp"])
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
        """
        Converts a DataFrame containing stock data into numpy arrays for price and technical indicators.

        Args:
            df (pd.DataFrame): DataFrame containing stock data with columns 'tic', 'timestamp', 'close', 
                            and the specified technical indicators.
            tech_indicator_list (list): List of technical indicator column names to include.
            if_vix (bool, optional): Whether to include VIX data (not applicable for Interactive Brokers). Defaults to False.

        Returns:
            tuple: A tuple containing:
                - price_array (np.ndarray): 2D numpy array with rows as timestamps and columns as stock tickers, 
                                            containing the close prices.
                - tech_array (np.ndarray): 2D numpy array with the same shape as price_array, containing the 
                                        values of the specified technical indicators.
        """

        print("Converting DataFrame to arrays...")
        print("DataFrame columns:", df.columns)
        print("DataFrame head:")
        print(df.head().to_markdown(index=False,numalign="left", stralign="left"))  # Formatted output

        # Ensure 'timestamp' is datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Group by 'tic' and 'timestamp' to ensure unique combinations
        # and select the last non-null values
        df_merged_data = (
            df.sort_values(by=["tic", "timestamp"])  # Sort for consistent aggregation
            .groupby(["tic", "timestamp"])
            .last()
            .reset_index()
            .dropna()
        )  # Drop rows with NaN values (typically first rows)

        # Pivot for price and technical indicators
        price_array = df_merged_data.pivot(index='timestamp', columns='tic', values='close').values
        tech_array = df_merged_data.pivot(index='timestamp', columns='tic', values=tech_indicator_list).values

        # Convert arrays to the correct data type to mitigate potential errors with DRL libraries.
        price_array = price_array.astype(np.float32)
        tech_array = tech_array.astype(np.float32)

        # Print shapes for debugging (optional)
        print("Price array shape:", price_array.shape)
        print("Tech array shape:", tech_array.shape)
        return price_array, tech_array


async def main():
    ib_processor = IBProcessor(IB_GATEWAY_HOST='127.0.0.1', IB_GATEWAY_PORT=7497)
    ib_processor.connect()  # Ensure connection is established before downloading data
    
    try:
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
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if str(e).startswith("This event loop is already running"):
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())

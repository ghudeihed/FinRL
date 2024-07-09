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
        print(f"tic_list: {tic_list}")
        
        times = pd.date_range('2023-01-01', '2023-12-31', freq='D').tz_localize('America/New_York')
        print(f"times: {times}")
        
        future_results = []
        for tic in tic_list:
            tmp_df = pd.DataFrame(index=times)
            tic_df = df[df['tic'] == tic].set_index('timestamp')
            tmp_df = tmp_df.join(tic_df, how='left').ffill().fillna(0)
            tmp_df['tic'] = tic
            future_results.append(tmp_df.reset_index().rename(columns={'index': 'timestamp'}))
        
        new_df = pd.concat(future_results).reset_index(drop=True)
        print(f"new_df: {new_df.head()}")
        
        # Ensure 'timestamp' is in datetime format and localized
        if 'timestamp' in new_df.columns:
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            if new_df['timestamp'].dt.tz is None:
                new_df['timestamp'] = new_df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
            else:
                new_df['timestamp'] = new_df['timestamp'].dt.tz_convert('America/New_York')
        
        # If 'level_0' exists, replace 'timestamp' values with 'level_0' to ensure uniqueness
        if 'level_0' in new_df.columns:
            new_df['timestamp'] = pd.to_datetime(new_df['level_0'], unit='s')
            if new_df['timestamp'].dt.tz is None:
                new_df['timestamp'] = new_df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
            else:
                new_df['timestamp'] = new_df['timestamp'].dt.tz_convert('America/New_York')
            new_df = new_df.drop(columns=['level_0'])
        
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
                    temp_indicator = stock[stock.tic == tic][[indicator]]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = tic
                    temp_indicator["timestamp"] = stock[stock.tic == tic].index
                    temp_indicator["timestamp"] = pd.to_datetime(temp_indicator["timestamp"])
                    # Ensure timestamp is in the correct time zone
                    if temp_indicator['timestamp'].dt.tz is None:
                        temp_indicator['timestamp'] = temp_indicator['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
                    else:
                        temp_indicator['timestamp'] = temp_indicator['timestamp'].dt.tz_convert('America/New_York')
                    
                    print(f"temp_indicator: {temp_indicator.head()}")  # Debug print
                    indicator_df = pd.concat([indicator_df, temp_indicator], axis=0, ignore_index=True)
                except Exception as e:
                    print(e)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
                
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

    def calculate_turbulence(self, data, time_period=252):
        print("Calculating turbulence")
        print(f"Columns: {data.columns}")
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        df_price_pivot = df_price_pivot.pct_change()

        # Ensure no duplicate timestamps
        df_price_pivot = df_price_pivot[~df_price_pivot.index.duplicated(keep='first')]

        unique_date = df.timestamp.unique()
        start = time_period
        turbulence_index = [0] * start
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            if filtered_hist_price.shape[1] == 0:
                turbulence_temp = 0
            else:
                cov_temp = filtered_hist_price.cov()
                current_temp = current_price[[x for x in filtered_hist_price.columns]] - np.mean(
                    filtered_hist_price, axis=0
                )
                # Regularize the covariance matrix
                cov_temp += np.eye(cov_temp.shape[0]) * 1e-6
                try:
                    temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                        current_temp.values.T
                    )
                    if temp > 0:
                        count += 1
                        if count > 2:
                            turbulence_temp = temp[0][0]
                        else:
                            turbulence_temp = 0
                    else:
                        turbulence_temp = 0
                except np.linalg.LinAlgError as e:
                    print(f"LinAlgError: {e}")
                    turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        print("Finished calculating turbulence")
        return turbulence_index

    def add_turbulence(self, data, time_period=252):
        print("Add turbulence")
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool=False):
        print("Converting DataFrame to arrays...")
        print("DataFrame columns:", df.columns)
        print("DataFrame head:")
        print(df.head().to_markdown(index=False,numalign="left", stralign="left"))  # Formatted output

        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIX"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        # Print shapes for debugging (optional)
        print("Price array shape:", price_array.shape)
        print("Tech array shape:", tech_array.shape)
        print("Turbulence array shape:", turbulence_array.shape)
        return price_array, tech_array, turbulence_array

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
        
        data_with_turbulence = ib_processor.add_turbulence(data_with_vix)
        print("Final DataFrame columns:", data_with_turbulence.columns)
        print("Final DataFrame head:")
        print(data_with_turbulence.head())
        
        price_array, tech_array, turbulence_array = ib_processor.df_to_array(data_with_turbulence, tech_indicator_list=['macd', 'rsi_30'])
        print("Price array shape:", price_array.shape)
        print("Tech array shape:", tech_array.shape)
        print("Turbulence array shape:", turbulence_array.shape)
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

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

class FeatureFactory:
       """
       A module to process raw OHLCV data into a normalized feature set 
       with heuristic signals for Reinforcement Learning agents.
       """

       def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=20, donchian_period=20, normalize=True, strategy='sma'):
              self.rsi_period = rsi_period
              self.donchian_period = donchian_period
              self.macd_fast = macd_fast
              self.macd_slow = macd_slow
              self.macd_signal = macd_signal
              self.sma_period = sma_period
              self.scaler = MinMaxScaler()
              self.need_normalization = normalize
              self.strategy = strategy

       def add_technical_indicators(self, df):
              """Adds RSI, MACD, and SMA to the dataframe."""
              # Ensure column names are lowercase for pandas_ta compatibility if needed
              df.ta.rsi(length=self.rsi_period, append=True)
              df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)
              df.ta.sma(length=self.sma_period, append=True)
              df.ta.ema(length=self.macd_slow, append=True)  # EMA for trend direction
              df.ta.ppo(length=self.macd_slow, append=True)
              df.ta.donchian(lower_length=self.donchian_period, upper_length=self.donchian_period, append=True)
              return df

       def generate_heuristic_signals(self, df):
              """
              Sub-module: Generates 'Buy' and 'Sell' signals based on trend-following logic:
              - Buy: Price > SMA AND RSI < 70 (Not overbought)
              - Sell: Price < SMA OR RSI > 70
              """
              # Close price column name might vary; assuming 'Close'
              close_col = 'close'
              sma_col = f'SMA_{self.sma_period}'
              rsi_col = f'RSI_{self.rsi_period}'

              df['signal_buy'] = np.where((df[close_col] > df[sma_col]) & (df[rsi_col] < 70), 1, 0)
              df['signal_sell'] = np.where((df[close_col] < df[sma_col]) | (df[rsi_col] > 80), 1, 0)
              
              return df

       def generate_donchian_signals(self, df):
              """
              Generates 'Buy' and 'Sell' signals based on the Donchian Channel:
              - Buy: Close price crosses above the upper channel.
              - Sell: Close price crosses below the lower channel.
              """
              close_col = 'close'
              donchian_upper_col = f'DCU_{self.donchian_period}_{self.donchian_period}'
              donchian_lower_col = f'DCL_{self.donchian_period}_{self.donchian_period}'

              df['signal_buy'] = np.where(df[close_col] > df[donchian_upper_col].shift(1), 1, 0)
              df['signal_sell'] = np.where(df[close_col] < df[donchian_lower_col].shift(1), 1, 0)
              
              return df

       def normalize_data(self, df):
              """Normalizes technical features to a 0-1 range for RL stability."""
              
              # ---------------------------------------------------------
              # STEP 1: Save the RAW price before scaling
              # We will use this column in the Environment for calculating 
              # real Net Worth and share quantities.
              # ---------------------------------------------------------
              df['raw_close'] = df['close'] 

              # List of columns to normalize (indicators)
              cols_to_scale = [
                     f'RSI_{self.rsi_period}', 
                     f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}', 
                     f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
                     f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
                     f'SMA_{self.sma_period}',
                     f'DCU_{self.donchian_period}_{self.donchian_period}',
                     f'DCL_{self.donchian_period}_{self.donchian_period}',
                     f'DCM_{self.donchian_period}_{self.donchian_period}',
              ]
              
              # Adding OHLCV to scaling for relative price movements
              # Note: 'close' is still here, so the Agent sees the NORMALIZED version
              ohlcv_cols = ['open', 'high', 'low', 'close', 'Trading_Volume']
              target_cols = ohlcv_cols + cols_to_scale
              
              # STEP 2: Scale the target columns (including 'close')
              df[target_cols] = self.scaler.fit_transform(df[target_cols])
              
              return df

       def process(self, raw_df):
              """
              Main pipeline: Cleans, calculates, signals, and normalizes.
              """
              df = raw_df.copy()
              
              # 1. Setup Index
              if 'date' in df.columns:
                     df['date'] = pd.to_datetime(df['date'])
                     df.set_index('date', inplace=True)
              df.sort_index(inplace=True)
              df['low'] = df['min']
              df['high'] = df['max']
              # 2. Add Indicators
              df = self.add_technical_indicators(df)
              
              # 3. Add Heuristic Signals
              if self.strategy == 'sma':
                     df = self.generate_heuristic_signals(df)
              elif self.strategy == 'donchian':
                     df = self.generate_donchian_signals(df)
              
              # 4. Clean up (Remove NaNs created by lagging indicators)
              df.dropna(inplace=True)
              
              # 5. Normalize
              processed_df = df
              if self.need_normalization:
                     processed_df = self.normalize_data(df)
              
              return processed_df
       

       def process_and_split(self, raw_df, train_ratio=0.8):
              """
              Processes data, splits into train/test (80/20 default), and normalizes separately
              to prevent data leakage.
              """
              df = raw_df.copy()
              
              # 1. Setup Index
              if 'date' in df.columns:
                     df['date'] = pd.to_datetime(df['date'])
                     df.set_index('date', inplace=True)
              df.sort_index(inplace=True)
              df['low'] = df['min']
              df['high'] = df['max']
              # 2. Add Indicators
              df = self.add_technical_indicators(df)
              
              # 3. Add Heuristic Signals
              if self.strategy == 'sma':
                     df = self.generate_heuristic_signals(df)
              elif self.strategy == 'donchian':
                     df = self.generate_donchian_signals(df)
              
              # 4. Clean up
              df.dropna(inplace=True)
              
              # 5. Split
              split_idx = int(len(df) * train_ratio)
              train_df = df.iloc[:split_idx].copy()
              test_df = df.iloc[split_idx:].copy()
              
              # 6. Normalize separately
              train_df, test_df = self._normalize_split(train_df, test_df)
              
              return train_df, test_df

       def _normalize_split(self, train_df, test_df):
              """Normalizes train and test sets using the scaler fitted on train set."""
              # Save raw close
              train_df['raw_close'] = train_df['close']
              test_df['raw_close'] = test_df['close']

              cols_to_scale = [
                     f'RSI_{self.rsi_period}', 
                     f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}', 
                     f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
                     f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
                     f'SMA_{self.sma_period}',
                     f'DCU_{self.donchian_period}_{self.donchian_period}',
                     f'DCL_{self.donchian_period}_{self.donchian_period}',
                     f'DCM_{self.donchian_period}_{self.donchian_period}',
              ]
              
              ohlcv_cols = ['open', 'high', 'low', 'close', 'Trading_Volume']
              target_cols = ohlcv_cols + cols_to_scale
              
              # Fit on Train, Transform Train
              train_df[target_cols] = self.scaler.fit_transform(train_df[target_cols])
              
              # Transform Test
              test_df[target_cols] = self.scaler.transform(test_df[target_cols])
              
              return train_df, test_df

# Example Usage:
# if __name__ == "__main__":
#     data = pd.read_csv('your_stock_data.csv')
#     factory = FeatureFactory()
#     clean_df = factory.process(data)
#     print(clean_df.head())
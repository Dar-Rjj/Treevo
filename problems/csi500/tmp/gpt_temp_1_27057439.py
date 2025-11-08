import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility-Adjusted Momentum
    df['hist_vol_20'] = df['true_range'].rolling(window=20).mean()
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    df['vol_adj_momentum'] = df['momentum_20'] / df['hist_vol_20']
    
    # Volume-Price Divergence
    def linear_regression_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                slope = (window * np.sum(x*y) - np.sum(x) * np.sum(y)) / (window * np.sum(x**2) - np.sum(x)**2)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    df['price_slope_5'] = linear_regression_slope(df['close'], 5)
    df['volume_slope_5'] = linear_regression_slope(df['volume'], 5)
    df['volume_price_divergence'] = df['price_slope_5'] * df['volume_slope_5']
    
    # Intraday Strength Persistence
    df['intraday_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['intraday_persistence'] = df['intraday_strength'].rolling(window=5).std()
    df['intraday_strength_persistence'] = df['intraday_strength'] * (1 - df['intraday_persistence'])
    
    # Amplitude-Adjusted Reversal
    df['price_amplitude'] = (df['high'] - df['low']) / df['close']
    df['daily_return'] = df['close'] / df['prev_close'] - 1
    df['amplitude_adj_reversal'] = -df['daily_return'] * df['price_amplitude']
    
    # Volume Breakout
    df['volume_cluster'] = df['volume'].rolling(window=10).mean().shift(1).rolling(window=3).mean()
    df['breakout_ratio'] = df['volume'] / df['volume_cluster']
    df['volume_breakout'] = df['breakout_ratio'] * df['daily_return']
    
    # Efficiency Momentum
    df['net_change'] = abs(df['close'] - df['close'].shift(10))
    
    def total_movement(close_series, window):
        movements = []
        for i in range(len(close_series)):
            if i >= window:
                total = 0
                for j in range(i-window+1, i+1):
                    total += abs(close_series.iloc[j] - close_series.iloc[j-1])
                movements.append(total)
            else:
                movements.append(np.nan)
        return pd.Series(movements, index=close_series.index)
    
    df['total_movement'] = total_movement(df['close'], 10)
    df['efficiency_ratio'] = df['net_change'] / df['total_movement']
    df['raw_return_10'] = df['close'] / df['close'].shift(10) - 1
    df['efficiency_momentum'] = df['efficiency_ratio'] * df['raw_return_10']
    
    # Gap-Fill Signal
    df['morning_gap'] = df['open'] / df['prev_close'] - 1
    df['intraday_range'] = df['high'] - df['low']
    df['filling_degree'] = abs(df['close'] - df['open']) / df['intraday_range']
    df['gap_fill_signal'] = -df['morning_gap'] * df['filling_degree']
    
    # Pressure Accumulation
    df['buying_pressure'] = np.where(df['close'] > df['prev_close'], df['close'] - df['low'], 0)
    df['selling_pressure'] = np.where(df['close'] < df['prev_close'], df['high'] - df['close'], 0)
    df['net_pressure'] = (df['buying_pressure'] - df['selling_pressure']).rolling(window=5).sum()
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    df['pressure_accumulation'] = df['net_pressure'] / df['atr_5']
    
    # Combine all factors with equal weights
    factors = [
        'vol_adj_momentum', 'volume_price_divergence', 'intraday_strength_persistence',
        'amplitude_adj_reversal', 'volume_breakout', 'efficiency_momentum',
        'gap_fill_signal', 'pressure_accumulation'
    ]
    
    # Calculate final factor as simple average of normalized individual factors
    factor_df = df[factors].copy()
    for col in factors:
        factor_df[col] = (factor_df[col] - factor_df[col].mean()) / factor_df[col].std()
    
    final_factor = factor_df.mean(axis=1)
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'hist_vol_20', 
                   'momentum_20', 'price_slope_5', 'volume_slope_5', 'intraday_persistence',
                   'price_amplitude', 'daily_return', 'volume_cluster', 'breakout_ratio',
                   'net_change', 'total_movement', 'efficiency_ratio', 'raw_return_10',
                   'morning_gap', 'intraday_range', 'filling_degree', 'buying_pressure',
                   'selling_pressure', 'net_pressure', 'atr_5'] + factors
    
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    return final_factor

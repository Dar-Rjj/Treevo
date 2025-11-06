import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Efficiency Score
    df['intraday_gain_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['intraday_gain_ratio'] = df['intraday_gain_ratio'].fillna(0)
    
    # Directional consistency (3-day sign persistence)
    df['close_open_sign'] = np.sign(df['close'] - df['open'])
    df['dir_consistency'] = df['close_open_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: len(set(x)) == 1 if len(x) == 3 else 0, raw=True
    )
    df['efficiency_score'] = df['intraday_gain_ratio'] * df['dir_consistency']
    
    # Bidirectional Flow Pressure
    mid_price = (df['high'] + df['low']) / 2
    df['upside_pressure'] = np.where(df['close'] > mid_price, df['volume'], 0)
    df['downside_pressure'] = np.where(df['close'] < mid_price, df['volume'], 0)
    
    # Rolling flow pressure ratios
    df['upside_pressure_5d'] = df['upside_pressure'].rolling(window=5, min_periods=1).sum()
    df['downside_pressure_5d'] = df['downside_pressure'].rolling(window=5, min_periods=1).sum()
    df['flow_pressure_ratio'] = (df['upside_pressure_5d'] - df['downside_pressure_5d']) / (
        df['upside_pressure_5d'] + df['downside_pressure_5d']).replace(0, np.nan)
    df['flow_pressure_ratio'] = df['flow_pressure_ratio'].fillna(0)
    
    # Efficiency-Flow Momentum Divergence
    df['efficiency_momentum'] = df['efficiency_score'].rolling(window=5, min_periods=1).mean()
    df['efficiency_momentum_change'] = df['efficiency_momentum'].diff(periods=5)
    
    df['flow_momentum'] = df['flow_pressure_ratio'].rolling(window=5, min_periods=1).mean()
    df['flow_momentum_change'] = df['flow_momentum'].diff(periods=5)
    
    # Momentum divergence
    df['momentum_divergence'] = df['efficiency_momentum_change'] - df['flow_momentum_change']
    
    # Adaptive Signal Weighting
    # Convergence scoring (correlation between efficiency and flow trends)
    df['efficiency_trend'] = df['efficiency_score'].rolling(window=5, min_periods=1).mean()
    df['flow_trend'] = df['flow_pressure_ratio'].rolling(window=5, min_periods=1).mean()
    
    # Rolling correlation for convergence scoring
    def rolling_corr(x, y):
        if len(x) < 2:
            return 0
        return np.corrcoef(x, y)[0, 1] if not (np.std(x) == 0 or np.std(y) == 0) else 0
    
    df['convergence_score'] = pd.Series(
        [rolling_corr(df['efficiency_trend'].iloc[max(0, i-4):i+1].values, 
                      df['flow_trend'].iloc[max(0, i-4):i+1].values) 
         for i in range(len(df))],
        index=df.index
    ).fillna(0)
    
    # Volume-intensity adjusted signal amplification
    df['volume_intensity'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
    df['signal_amplification'] = 1 + np.abs(df['volume_intensity'] - 1)
    
    # Final factor calculation
    df['factor'] = (df['momentum_divergence'] * 
                   (1 + df['convergence_score']) * 
                   df['signal_amplification'])
    
    return df['factor']

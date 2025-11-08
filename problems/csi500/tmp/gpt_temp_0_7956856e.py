import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Volatility Asymmetry
    # Upside Volatility: avg(max(0, High_t - Close_t)) over 10 days
    data['upside_move'] = np.maximum(0, data['high'] - data['close'])
    data['upside_vol'] = data['upside_move'].rolling(window=10, min_periods=5).mean()
    
    # Downside Volatility: avg(max(0, Close_t - Low_t)) over 10 days
    data['downside_move'] = np.maximum(0, data['close'] - data['low'])
    data['downside_vol'] = data['downside_move'].rolling(window=10, min_periods=5).mean()
    
    # Asymmetry Ratio: Upside Volatility / Downside Volatility
    data['asymmetry_ratio'] = data['upside_vol'] / data['downside_vol']
    data['asymmetry_ratio'] = data['asymmetry_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Price-Volume Efficiency
    # Intraday Efficiency: (Close_t - Low_t) / (High_t - Low_t)
    data['intraday_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Percentile: rank(Volume_t) over 20 days
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Efficiency Signal: Intraday Efficiency × Volume Percentile
    data['efficiency_signal'] = data['intraday_efficiency'] * data['volume_rank']
    
    # Momentum Quality
    # Directional Persistence: consecutive same-sign returns over 5 days
    def calc_directional_persistence(returns_series):
        if len(returns_series) < 5:
            return np.nan
        signs = np.sign(returns_series[-5:])
        persistence = 0
        for i in range(1, 5):
            if signs[i] == signs[i-1] and signs[i] != 0:
                persistence += 1
        return persistence
    
    data['directional_persistence'] = data['returns'].rolling(window=5, min_periods=5).apply(
        calc_directional_persistence, raw=False
    )
    
    # Return-to-Volatility: 5-day return / 5-day volatility
    data['five_day_return'] = data['close'].pct_change(periods=5)
    data['five_day_vol'] = data['returns'].rolling(window=5, min_periods=5).std()
    data['return_to_vol'] = data['five_day_return'] / data['five_day_vol']
    data['return_to_vol'] = data['return_to_vol'].replace([np.inf, -np.inf], np.nan)
    
    # Quality Signal: Directional Persistence × Return-to-Volatility
    data['quality_signal'] = data['directional_persistence'] * data['return_to_vol']
    
    # Composite Alpha
    # Base Factor: Asymmetry Ratio × Efficiency Signal
    data['base_factor'] = data['asymmetry_ratio'] * data['efficiency_signal']
    
    # Final Alpha: Base Factor × Quality Signal
    data['alpha'] = data['base_factor'] * data['quality_signal']
    
    # Return the alpha series with the same index as input
    return data['alpha']

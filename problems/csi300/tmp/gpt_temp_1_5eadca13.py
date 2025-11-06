import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Fractal Price Dynamics
    data['short_term_range'] = (data['high'] - data['low']) / data['close']
    data['medium_term_range'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close'].shift(4)
    data['fractal_compression'] = data['short_term_range'] / data['medium_term_range']
    
    # Volume Asymmetry
    data['up_day_volume'] = np.where(data['close'] > data['close'].shift(1), data['volume'], np.nan)
    data['down_day_volume'] = np.where(data['close'] < data['close'].shift(1), data['volume'], np.nan)
    
    up_volume_rolling = data['up_day_volume'].rolling(window=5, min_periods=1).mean()
    down_volume_rolling = data['down_day_volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = up_volume_rolling / down_volume_rolling
    
    # Price-Volume Alignment
    data['fractal_expansion'] = (data['fractal_compression'] > 1.2).astype(int)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(4)
    
    price_change_sign = np.sign(data['close'] - data['close'].shift(1))
    volume_momentum_sign = np.sign(data['volume_momentum'] - 1)
    data['divergence_signal'] = price_change_sign * volume_momentum_sign
    
    # Microstructure Dynamics
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = np.abs(data['close'] - data['mid_price']) / data['mid_price']
    
    # Calculate average trade size
    data['avg_trade_size'] = data['amount'] / data['volume']
    avg_trade_size_4d = data['avg_trade_size'].shift(4)
    
    # Trade Size Skew calculation
    def calculate_trade_size_skew(row, window=3):
        if pd.isna(avg_trade_size_4d[row.name]):
            return 0
        current_idx = data.index.get_loc(row.name)
        if current_idx < window:
            return 0
        
        large_trades = 0
        small_trades = 0
        for i in range(current_idx - window, current_idx):
            if not pd.isna(data.iloc[i]['avg_trade_size']) and not pd.isna(avg_trade_size_4d[row.name]):
                if data.iloc[i]['avg_trade_size'] > 2 * avg_trade_size_4d[row.name]:
                    large_trades += 1
                elif data.iloc[i]['avg_trade_size'] < 0.5 * avg_trade_size_4d[row.name]:
                    small_trades += 1
        
        return (large_trades / window) - (small_trades / window)
    
    data['trade_size_skew'] = data.apply(calculate_trade_size_skew, axis=1)
    
    # Bid-Ask Pressure
    high_low_range = data['high'] - data['low']
    data['bid_ask_pressure'] = ((data['close'] - data['low']) / high_low_range * data['volume'] - 
                               (data['high'] - data['close']) / high_low_range * data['volume'])
    
    # Momentum Patterns
    data['short_term_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(4) - 1
    data['momentum_compression'] = data['short_term_momentum'] / data['medium_term_momentum']
    
    # Efficiency Metrics
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['efficiency_compression'] = (data['intraday_efficiency'] / 
                                    np.abs(data['open'] - data['close'].shift(1)) * 
                                    (data['high'] - data['low']))
    
    # Alpha Synthesis
    data['fractal_volume_alpha'] = (data['divergence_signal'] * 
                                   data['volume_momentum'] * 
                                   data['fractal_compression'])
    
    data['microstructure_alpha'] = data['bid_ask_pressure'] * data['trade_size_skew']
    
    data['momentum_alpha'] = data['momentum_compression'] * data['volume_ratio']
    
    data['efficiency_alpha'] = data['volume_efficiency'] * data['efficiency_compression']
    
    # Composite Alpha
    data['composite_alpha'] = (data['fractal_volume_alpha'] + 
                              data['microstructure_alpha'] + 
                              data['momentum_alpha'] + 
                              data['efficiency_alpha'])
    
    return data['composite_alpha']

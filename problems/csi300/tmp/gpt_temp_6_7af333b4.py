import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday returns and range
    data['intraday_return'] = data['high'] / data['low'] - 1
    data['close_return'] = data['close'].pct_change()
    
    # Price Momentum Components
    # Intraday momentum - 5-day rate of change of intraday range
    data['intraday_roc_5d'] = data['intraday_return'] / data['intraday_return'].shift(5) - 1
    
    # Close price momentum
    data['close_momentum_5d'] = data['close_return'].rolling(window=5).sum()
    data['close_volatility_10d'] = data['close_return'].rolling(window=10).std()
    
    # Volume Dynamics Analysis
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_20d'] = data['volume'].rolling(window=20).mean()
    data['volume_trend_ratio'] = data['volume_ma_5d'] / data['volume_ma_20d']
    
    # Volume momentum signals
    data['volume_change_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_trend_slope_10d'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    
    # Volatility Normalization
    data['intraday_volatility_20d'] = data['intraday_return'].rolling(window=20).std()
    
    # Adjust momentum components by volatility
    data['vol_adj_intraday_momentum'] = data['intraday_roc_5d'] / data['intraday_volatility_20d']
    data['vol_adj_close_momentum'] = data['close_momentum_5d'] / data['close_volatility_10d']
    
    # Combined price momentum
    data['combined_price_momentum'] = (data['vol_adj_intraday_momentum'] + data['vol_adj_close_momentum']) / 2
    
    # Volume momentum direction
    data['volume_momentum_direction'] = np.sign(data['volume_trend_slope_10d'])
    
    # Divergence Detection
    data['price_momentum_direction'] = np.sign(data['combined_price_momentum'])
    
    # Divergence signal: positive when price and volume momentum move in opposite directions
    data['divergence_direction'] = data['price_momentum_direction'] * data['volume_momentum_direction'] * -1
    
    # Divergence magnitude based on the absolute difference between normalized signals
    price_momentum_norm = data['combined_price_momentum'] / data['combined_price_momentum'].rolling(window=20).std()
    volume_momentum_norm = data['volume_trend_slope_10d'] / data['volume_trend_slope_10d'].rolling(window=20).std()
    data['divergence_magnitude'] = abs(price_momentum_norm - volume_momentum_norm)
    
    # Factor Synthesis
    # Primary signal: volatility-adjusted price momentum amplified by volume dynamics
    primary_signal = (data['combined_price_momentum'] * 
                     data['volume_trend_ratio'] * 
                     (1 + data['volume_trend_slope_10d'] / 100))
    
    # Final factor: apply divergence direction and scale by magnitude
    final_factor = primary_signal * data['divergence_direction'] * data['divergence_magnitude']
    
    return final_factor

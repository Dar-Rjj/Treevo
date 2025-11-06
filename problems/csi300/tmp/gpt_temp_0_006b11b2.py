import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-scaled momentum convergence with volume validation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    # Short-term reversal (3-day)
    data['daily_ret'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['momentum_3d'] = (data['close'] / data['close'].shift(3) - 1).fillna(0)
    
    # Medium-term momentum (10-day)
    data['momentum_10d'] = (data['close'] / data['close'].shift(10) - 1).fillna(0)
    
    # Momentum Divergence
    data['momentum_divergence'] = data['momentum_10d'] - data['momentum_3d']
    
    # Dynamic Volatility Scaling
    # Intraday Range Volatility
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['range_vol_5d'] = data['intraday_range'].rolling(window=5, min_periods=3).mean()
    
    # Close-to-Close Volatility
    data['abs_return'] = abs(data['daily_ret'])
    data['close_vol_5d'] = data['abs_return'].rolling(window=5, min_periods=3).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    
    # Combined Volatility Measure
    epsilon = 0.0001
    data['combined_vol'] = ((data['range_vol_5d'] + data['close_vol_5d']) / 2 + epsilon).fillna(epsilon)
    
    # Volume-Weighted Factor Construction
    # Volume Trend Analysis
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5']
    
    # Volume-Price Confirmation
    def rolling_corr(x):
        if len(x) < 5:
            return 0
        prices = x[:, 0]
        volumes = x[:, 1]
        if np.std(prices) == 0 or np.std(volumes) == 0:
            return 0
        return np.corrcoef(prices, volumes)[0, 1]
    
    # Prepare data for correlation calculation
    price_changes = data['close'].pct_change().fillna(0)
    combined_data = np.column_stack([price_changes.values, data['volume_ratio'].values])
    
    # Calculate rolling correlation
    data['volume_price_corr'] = pd.Series(
        [rolling_corr(combined_data[i-4:i+1]) if i >= 4 else 0 
         for i in range(len(combined_data))],
        index=data.index
    ).fillna(0)
    
    # Final Factor Construction
    # Divide momentum divergence by volatility measure
    volatility_scaled_momentum = data['momentum_divergence'] / data['combined_vol']
    
    # Multiply by volume confirmation score (absolute correlation)
    volume_confirmation = abs(data['volume_price_corr'])
    
    # Apply sign based on medium-term momentum direction
    momentum_sign = np.sign(data['momentum_10d'])
    
    # Combine all components
    factor = volatility_scaled_momentum * volume_confirmation * momentum_sign
    
    # Clean and return the factor
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor

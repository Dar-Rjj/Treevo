import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility-adjusted momentum, 
    volume-price synchronization, intraday strength, and price level analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor components
    factors = pd.DataFrame(index=data.index)
    
    # 1. Volatility-Adjusted Momentum Factor
    # Calculate momentum
    factors['mom_short'] = data['close'] / data['close'].shift(5) - 1
    factors['mom_medium'] = data['close'] / data['close'].shift(10) - 1
    
    # Calculate volatility measures
    factors['daily_range'] = (data['high'] - data['low']) / data['close']
    factors['rolling_vol'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    
    # Volatility-adjusted momentum
    vol_threshold = factors['rolling_vol'].quantile(0.7)
    factors['vol_adj_momentum'] = np.where(
        factors['rolling_vol'] > vol_threshold,
        factors['mom_short'] / (factors['rolling_vol'] + 1e-8),
        factors['mom_short'] * factors['rolling_vol']
    )
    
    # 2. Volume-Price Synchronization Factor
    # Volume patterns
    factors['vol_momentum'] = data['volume'] / data['volume'].shift(5)
    factors['vol_stability'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Price-volume correlation
    price_change = data['close'].pct_change()
    volume_change = data['volume'].pct_change()
    
    factors['price_vol_sync'] = np.where(
        np.sign(price_change) == np.sign(volume_change),
        np.abs(price_change) * np.abs(volume_change),
        -np.abs(price_change) * np.abs(volume_change)
    )
    
    # Volume confirmation score
    price_up = price_change > 0
    volume_up = volume_change > 0
    price_down = price_change < 0
    volume_down = volume_change < 0
    
    factors['volume_confirmation'] = np.where(
        (price_up & volume_up) | (price_down & volume_down),
        np.abs(price_change) * factors['vol_momentum'],
        -np.abs(price_change) * factors['vol_momentum']
    )
    
    # 3. Intraday Strength Factor
    # Intraday performance
    factors['opening_strength'] = data['open'] / data['close'].shift(1) - 1
    factors['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume-adjusted intraday moves
    vol_quantile = data['volume'].rolling(window=20).apply(lambda x: x.quantile(0.7))
    high_volume = data['volume'] > vol_quantile
    
    factors['intraday_volume_strength'] = np.where(
        high_volume,
        factors['opening_strength'] * factors['range_efficiency'],
        factors['opening_strength'] * factors['range_efficiency'] * 0.5
    )
    
    # Early vs late volume concentration (simplified)
    factors['volume_timing'] = data['amount'].rolling(window=3).mean() / data['amount'].rolling(window=10).mean()
    
    # 4. Price Level Factor
    # Relative price position
    factors['daily_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Weekly range position
    weekly_high = data['high'].rolling(window=5).max()
    weekly_low = data['low'].rolling(window=5).min()
    factors['weekly_position'] = (data['close'] - weekly_low) / (weekly_high - weekly_low + 1e-8)
    
    # Extreme price detection
    extreme_vol = data['volume'] > data['volume'].rolling(window=20).quantile(0.8)
    overbought = (factors['daily_position'] > 0.8) & extreme_vol
    oversold = (factors['daily_position'] < 0.2) & extreme_vol
    
    factors['extreme_signal'] = np.where(
        overbought, -1, np.where(oversold, 1, 0)
    ) * factors['weekly_position']
    
    # Combine all factors with equal weights
    factor_components = [
        'vol_adj_momentum',
        'price_vol_sync', 
        'volume_confirmation',
        'intraday_volume_strength',
        'daily_position',
        'weekly_position',
        'extreme_signal'
    ]
    
    # Normalize and combine
    for col in factor_components:
        if col in factors.columns:
            factors[f'{col}_norm'] = (factors[col] - factors[col].mean()) / (factors[col].std() + 1e-8)
    
    # Final alpha factor (equal weighted combination)
    normalized_cols = [f'{col}_norm' for col in factor_components if f'{col}_norm' in factors.columns]
    final_factor = factors[normalized_cols].mean(axis=1)
    
    return final_factor

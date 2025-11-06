import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Momentum Divergence Factor
    Analyzes the relationship between price momentum and volume momentum to identify divergence patterns
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Price Momentum Analysis
    # Short-term Price Momentum
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Price Range Dynamics
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_momentum_5'] = data['daily_range'] / data['daily_range'].shift(5) - 1
    
    # Range expansion/contraction
    data['range_expansion'] = data['daily_range'] > data['daily_range'].rolling(window=10).mean()
    
    # Volatility Context
    data['price_change_magnitude'] = abs(data['close'].pct_change())
    data['recent_volatility'] = data['close'].pct_change().rolling(window=10).std()
    data['historical_volatility'] = data['close'].pct_change().rolling(window=50).std()
    data['volatility_ratio'] = data['recent_volatility'] / data['historical_volatility']
    
    # Volume Momentum Analysis
    # Short-term Volume Momentum
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Volume Pattern Recognition
    data['volume_trend_5'] = data['volume'].rolling(window=5).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[0] and all(x.diff().dropna() >= 0)) 
        else (-1 if (x.iloc[-1] < x.iloc[0] and all(x.diff().dropna() <= 0)) else 0), 
        raw=False
    )
    
    # Volume spike identification
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['volume_spike'] = data['volume_zscore'] > 2
    
    # Volume Rate of Change
    data['daily_volume_change'] = data['volume'].pct_change()
    data['volume_acceleration'] = data['daily_volume_change'].diff()
    
    # Divergence Signal Generation
    # Directional Alignment Assessment
    data['price_volume_correlation_10'] = data['close'].pct_change().rolling(window=10).corr(data['volume'].pct_change())
    
    # Trend consistency
    data['price_trend_strength'] = (
        np.sign(data['price_momentum_5']) + 
        np.sign(data['price_momentum_10']) + 
        np.sign(data['price_momentum_20'])
    ) / 3
    
    data['volume_trend_strength'] = (
        np.sign(data['volume_momentum_5']) + 
        np.sign(data['volume_momentum_10']) + 
        np.sign(data['volume_momentum_20'])
    ) / 3
    
    # Divergence Pattern Classification
    # Strong Bullish Divergence (Price up, Volume down)
    bullish_divergence = (
        (data['price_momentum_5'] > 0) & 
        (data['volume_momentum_5'] < 0) & 
        (data['price_trend_strength'] > 0.5) & 
        (data['volume_trend_strength'] < -0.5)
    )
    
    # Strong Bearish Divergence (Price down, Volume up)
    bearish_divergence = (
        (data['price_momentum_5'] < 0) & 
        (data['volume_momentum_5'] > 0) & 
        (data['price_trend_strength'] < -0.5) & 
        (data['volume_trend_strength'] > 0.5)
    )
    
    # Confirmed Bullish (Price up, Volume up)
    confirmed_bullish = (
        (data['price_momentum_5'] > 0) & 
        (data['volume_momentum_5'] > 0) & 
        (data['price_trend_strength'] > 0.3) & 
        (data['volume_trend_strength'] > 0.3)
    )
    
    # Confirmed Bearish (Price down, Volume down)
    confirmed_bearish = (
        (data['price_momentum_5'] < 0) & 
        (data['volume_momentum_5'] < 0) & 
        (data['price_trend_strength'] < -0.3) & 
        (data['volume_trend_strength'] < -0.3)
    )
    
    # Signal Strength Quantification
    # Divergence magnitude
    data['divergence_magnitude'] = (
        (data['price_momentum_5'] - data['volume_momentum_5']) * 
        data['price_volume_correlation_10'].abs()
    )
    
    # Pattern persistence
    data['bullish_divergence_persistence'] = bullish_divergence.rolling(window=3).sum()
    data['bearish_divergence_persistence'] = bearish_divergence.rolling(window=3).sum()
    
    # Multi-timeframe confirmation
    data['multi_timeframe_confirmation'] = (
        np.sign(data['price_momentum_5']) * 0.4 + 
        np.sign(data['price_momentum_10']) * 0.3 + 
        np.sign(data['price_momentum_20']) * 0.3
    )
    
    # Final factor calculation
    factor = pd.Series(index=data.index, dtype=float)
    
    # Assign scores based on divergence patterns
    factor[bullish_divergence] = (
        data['divergence_magnitude'] * 1.5 + 
        data['bullish_divergence_persistence'] * 0.2 + 
        data['multi_timeframe_confirmation'] * 0.3
    )[bullish_divergence]
    
    factor[bearish_divergence] = (
        data['divergence_magnitude'] * (-1.5) + 
        data['bearish_divergence_persistence'] * (-0.2) + 
        data['multi_timeframe_confirmation'] * 0.3
    )[bearish_divergence]
    
    factor[confirmed_bullish] = (
        data['price_momentum_5'] * 0.8 + 
        data['volume_momentum_5'] * 0.2 + 
        data['multi_timeframe_confirmation'] * 0.4
    )[confirmed_bullish]
    
    factor[confirmed_bearish] = (
        data['price_momentum_5'] * 0.8 + 
        data['volume_momentum_5'] * 0.2 + 
        data['multi_timeframe_confirmation'] * 0.4
    )[confirmed_bearish]
    
    # Fill neutral/weak signals with weighted combination
    neutral_mask = factor.isna()
    factor[neutral_mask] = (
        data['price_momentum_5'] * 0.6 + 
        data['volume_momentum_5'] * 0.2 + 
        data['divergence_magnitude'] * 0.2
    )[neutral_mask]
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=50).mean()) / factor.rolling(window=50).std()
    
    return factor

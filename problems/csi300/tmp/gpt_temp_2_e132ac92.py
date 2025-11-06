import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Quality with Volatility-Regime Volume Verification factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Momentum Quality Assessment
    # Momentum Purity - consistency of daily returns (5-day return volatility)
    data['return_vol_5d'] = data['returns'].rolling(window=5).std()
    
    # Momentum smoothness - ratio of 3-day to 1-day momentum volatility
    data['momentum_3d'] = data['close'].pct_change(periods=3)
    data['momentum_1d'] = data['returns']
    data['momentum_smoothness'] = data['momentum_3d'].rolling(window=5).std() / (data['momentum_1d'].rolling(window=5).std() + 1e-8)
    
    # Momentum persistence - consecutive days with same direction
    data['return_sign'] = np.sign(data['returns'])
    data['persistence_count'] = 0
    for i in range(1, len(data)):
        if data['return_sign'].iloc[i] == data['return_sign'].iloc[i-1]:
            data['persistence_count'].iloc[i] = data['persistence_count'].iloc[i-1] + 1
        else:
            data['persistence_count'].iloc[i] = 0
    
    # Momentum Quality Score
    data['momentum_quality'] = (
        (1 / (data['return_vol_5d'] + 1e-8)) +  # Higher consistency = better quality
        data['momentum_smoothness'] +  # Smoother momentum = better quality
        data['persistence_count']  # Higher persistence = better quality
    )
    
    # Volatility Regime Analysis
    # Daily price range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # 10-day price range magnitude
    data['range_magnitude_10d'] = data['daily_range'].rolling(window=10).mean()
    
    # Volatility persistence - autocorrelation of daily ranges
    data['range_autocorr'] = data['daily_range'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Regime shifts - changes in range magnitude
    data['range_magnitude_change'] = data['range_magnitude_10d'].pct_change(periods=5)
    
    # Volatility Regime Classification
    data['volatility_regime'] = 0
    high_vol_threshold = data['range_magnitude_10d'].quantile(0.7)
    low_vol_threshold = data['range_magnitude_10d'].quantile(0.3)
    
    mask_trending = (data['range_magnitude_10d'] > high_vol_threshold) & (data['range_autocorr'] > 0.3)
    mask_mean_reverting = (data['range_magnitude_10d'] < low_vol_threshold) & (data['range_autocorr'] < 0.1)
    mask_transition = (abs(data['range_magnitude_change']) > 0.1)
    
    data.loc[mask_trending, 'volatility_regime'] = 1  # Trending regime
    data.loc[mask_mean_reverting, 'volatility_regime'] = -1  # Mean-reverting regime
    data.loc[mask_transition & (data['volatility_regime'] == 0), 'volatility_regime'] = 0.5  # Transition phase
    
    # Volume-Verified Momentum Signals
    # Volume momentum trend
    data['volume_3d_change'] = data['volume'].pct_change(periods=3)
    data['volume_trend'] = np.sign(data['volume_3d_change'])
    
    # Price-volume direction alignment
    data['price_volume_alignment'] = np.sign(data['returns']) * np.sign(data['volume_3d_change'])
    
    # Volume surge during quality momentum moves
    data['volume_surge'] = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # Volume divergence during poor quality moves
    data['volume_divergence'] = np.where(
        (np.sign(data['returns']) != np.sign(data['volume_3d_change'])) & 
        (data['momentum_quality'] < data['momentum_quality'].rolling(window=10).mean()),
        abs(data['volume_3d_change']), 0
    )
    
    # Generate Regime-Aware Signals
    # Strong breakout signal
    strong_breakout = (
        (data['momentum_quality'] > data['momentum_quality'].rolling(window=10).mean()) &  # High quality momentum
        (data['price_volume_alignment'] > 0) &  # Volume confirmation
        (data['volatility_regime'] == -1) &  # Low volatility regime
        (data['volume_surge'] > 1.2)  # Volume surge
    )
    
    # Reversal warning signal
    reversal_warning = (
        (data['momentum_quality'] < data['momentum_quality'].rolling(window=10).mean()) &  # Poor quality momentum
        (data['price_volume_alignment'] < 0) &  # Volume divergence
        (data['volatility_regime'] == 1) &  # High volatility regime
        (data['volume_divergence'] > 0.1)  # Significant divergence
    )
    
    # Trend development signal
    trend_development = (
        (data['momentum_quality'] > data['momentum_quality'].shift(1)) &  # Improving quality
        (data['volume_3d_change'] > 0) &  # Increasing volume
        (data['volatility_regime'] == 0.5)  # Changing volatility
    )
    
    # Combine signals into final factor
    factor = pd.Series(0.0, index=data.index)
    factor[strong_breakout] += 2.0
    factor[reversal_warning] -= 1.5
    factor[trend_development] += 1.0
    
    # Add momentum quality as base component
    normalized_quality = (data['momentum_quality'] - data['momentum_quality'].rolling(window=20).mean()) / (data['momentum_quality'].rolling(window=20).std() + 1e-8)
    factor += normalized_quality * 0.5
    
    # Add volatility regime adjustment
    regime_adjustment = data['volatility_regime'].replace({1: -0.3, -1: 0.5, 0.5: 0.1, 0: 0})
    factor += regime_adjustment
    
    # Add volume confirmation component
    volume_confirmation = np.where(
        data['price_volume_alignment'] > 0,
        data['volume_surge'] * 0.2,
        -data['volume_divergence'] * 0.3
    )
    factor += volume_confirmation
    
    # Fill NaN values
    factor = factor.fillna(0)
    
    return factor

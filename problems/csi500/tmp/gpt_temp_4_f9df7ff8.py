import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Volume-Price Divergence Momentum with Bidirectional Flow Detection
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Volume-Price Divergence Component
    # 1.1 Compute Price Momentum Acceleration
    momentum_slopes = []
    for i in range(len(data)):
        if i >= 2:
            # Calculate 3-day price momentum slope (t-2 to t)
            x = np.array([0, 1, 2])
            y = data['close'].iloc[i-2:i+1].values
            if len(y) == 3 and not np.any(np.isnan(y)):
                slope, _, _, _, _ = linregress(x, y)
                momentum_slopes.append(slope)
            else:
                momentum_slopes.append(np.nan)
        else:
            momentum_slopes.append(np.nan)
    
    data['momentum_slope'] = momentum_slopes
    
    # Compute Momentum Change Rate
    data['momentum_change'] = data['momentum_slope'] / data['momentum_slope'].shift(1)
    data['momentum_change'] = data['momentum_change'].replace([np.inf, -np.inf], np.nan)
    
    # 1.2 Calculate Volume Flow Divergence
    # Compute Directional Volume Intensity
    data['signed_volume_flow'] = data['volume'] * (data['close'] - data['open']) / (data['high'] - data['low'])
    data['signed_volume_flow'] = data['signed_volume_flow'].replace([np.inf, -np.inf], np.nan)
    
    # Apply 3-day exponential weighting
    alpha = 0.5
    data['weighted_volume_flow'] = data['signed_volume_flow'].ewm(alpha=alpha, adjust=False).mean()
    
    # Detect Volume-Price Divergence
    data['raw_divergence'] = data['momentum_change'] * data['weighted_volume_flow']
    data['divergence_asin'] = np.arcsinh(data['raw_divergence'])
    
    # Apply divergence magnitude scaling
    data['price_range'] = data['high'] - data['low']
    data['scaled_divergence'] = data['divergence_asin'] * data['price_range']
    
    # 2. Implement Bidirectional Flow Analysis
    # 2.1 Calculate Intraday Pressure Asymmetry
    data['opening_pressure'] = (data['open'] - data['low']) / (data['high'] - data['low']) - 0.5
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['net_flow'] = data['closing_pressure'] - data['opening_pressure']
    
    # 2.2 Apply Multi-timeframe Flow Confirmation
    # Calculate Short-term Flow Consistency
    data['flow_sign'] = np.sign(data['net_flow'])
    data['flow_consistency'] = (data['flow_sign'] == data['flow_sign'].shift(1)).astype(float)
    data['flow_magnitude_product'] = abs(data['net_flow'] * data['net_flow'].shift(1))
    data['flow_consistency_weighted'] = data['flow_consistency'] * data['flow_magnitude_product']
    
    # Detect Flow Reversal Patterns
    data['opening_sign'] = np.sign(data['opening_pressure'])
    data['closing_sign'] = np.sign(data['closing_pressure'])
    data['sign_difference'] = (data['opening_sign'] != data['closing_sign']).astype(float)
    data['pressure_diff'] = abs(data['closing_pressure'] - data['opening_pressure'])
    
    # Calculate 5-day ATR (t-5 to t-1)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_5'] = data['tr'].rolling(window=5, min_periods=3).mean().shift(1)
    
    data['reversal_intensity'] = data['sign_difference'] * data['pressure_diff'] * data['volume'] / data['atr_5']
    data['reversal_intensity'] = data['reversal_intensity'].replace([np.inf, -np.inf], np.nan)
    
    # 3. Combine Divergence with Flow Analysis
    # Multiply Volume-Price Divergence by Flow Consistency
    data['divergence_flow_combined'] = data['scaled_divergence'] * data['flow_consistency_weighted']
    
    # Apply Bidirectional Flow Modulation
    data['modulated_divergence'] = data['divergence_flow_combined'] * (1 + data['reversal_intensity'])
    
    # Adjust for Market Regime
    # Calculate recent volatility regime
    data['returns'] = data['close'].pct_change()
    data['vol_5'] = data['returns'].rolling(window=5, min_periods=3).std()
    data['vol_20_hist'] = data['returns'].rolling(window=20, min_periods=10).std().shift(5)
    
    data['vol_ratio'] = data['vol_5'] / data['vol_20_hist']
    data['vol_ratio'] = data['vol_ratio'].replace([np.inf, -np.inf], np.nan)
    data['regime_adjustment'] = np.sqrt(data['vol_ratio'])
    
    # 4. Generate Final Alpha Factor
    # Combine All Weighted Components
    data['raw_factor'] = data['modulated_divergence'] * data['regime_adjustment']
    
    # Apply directional consistency check
    data['price_momentum_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['factor_sign'] = np.sign(data['raw_factor'])
    
    # Invert factor if major contradiction detected (opposite signs for 2+ consecutive days)
    data['sign_agreement'] = data['price_momentum_sign'] * data['factor_sign']
    data['contradiction_count'] = (data['sign_agreement'] < 0).rolling(window=2, min_periods=2).sum()
    data['inversion_flag'] = (data['contradiction_count'] >= 2).astype(float)
    
    data['adjusted_factor'] = data['raw_factor'] * (1 - 2 * data['inversion_flag'])
    
    # Implement Adaptive Smoothing
    # Volatility-dependent smoothing window
    vol_threshold = data['vol_20_hist'].quantile(0.7)
    data['smoothing_window'] = np.where(data['vol_5'] > vol_threshold, 3, 5)
    
    # Apply rolling mean with adaptive window
    final_factor = []
    for i in range(len(data)):
        if i >= 4:  # Need enough data for smoothing
            window = int(data['smoothing_window'].iloc[i])
            start_idx = max(0, i - window + 1)
            smoothed_value = data['adjusted_factor'].iloc[start_idx:i+1].mean()
            final_factor.append(smoothed_value)
        else:
            final_factor.append(data['adjusted_factor'].iloc[i])
    
    result = pd.Series(final_factor, index=data.index, name='bidirectional_flow_divergence_momentum')
    
    return result

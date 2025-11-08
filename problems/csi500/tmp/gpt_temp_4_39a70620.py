import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Acceleration with Volume Divergence alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # 1. Detect Price Regime using Multi-Timeframe Momentum
    # Long-term trend (60-day)
    df['long_term_return'] = df['close'].pct_change(periods=60)
    df['bull_regime'] = df['long_term_return'] > 0
    
    # Medium-term momentum (20-day)
    df['medium_term_return'] = df['close'].pct_change(periods=20)
    df['momentum_strength'] = df['medium_term_return'].abs()
    
    # 2. Calculate Momentum Acceleration
    df['short_term_return'] = df['close'].pct_change(periods=5)
    
    # Avoid division by zero
    medium_term_abs = df['medium_term_return'].abs()
    medium_term_abs = medium_term_abs.replace(0, np.nan)
    
    # Calculate acceleration ratio
    df['acceleration_ratio'] = df['short_term_return'] / medium_term_abs
    
    # Apply regime-dependent scaling
    def regime_scaling(row):
        if pd.isna(row['acceleration_ratio']):
            return np.nan
        if row['bull_regime']:
            return row['acceleration_ratio']
        else:
            # In bear regime, invert for reversal effect
            return -row['acceleration_ratio']
    
    df['regime_scaled_acceleration'] = df.apply(regime_scaling, axis=1)
    
    # 3. Incorporate Volume Divergence
    # Calculate volume changes
    df['volume_change'] = df['volume'].pct_change()
    
    # Price-Volume Correlation (10-day rolling)
    df['price_volume_corr'] = df['returns'].rolling(window=10).corr(df['volume_change'])
    
    # Identify divergence (negative correlation)
    df['volume_divergence'] = df['price_volume_corr'] < 0
    
    # Volume Surge Indicator (5-day volume percentile vs past 20 days)
    df['volume_percentile'] = df['volume'].rolling(window=20).apply(
        lambda x: (x[-5:].mean() > x[:-5]).mean() if len(x) == 20 else np.nan
    )
    
    # Combine divergence and volume surge
    df['volume_signal'] = df['volume_divergence'].astype(int) * df['volume_percentile']
    
    # 4. Apply Volatility Scaling
    # Calculate regime-specific volatility
    df['bull_volatility'] = df['returns'].rolling(window=20).std()
    df['bear_volatility'] = df['returns'].rolling(window=10).std()
    
    def regime_volatility(row):
        if pd.isna(row['bull_volatility']) or pd.isna(row['bear_volatility']):
            return np.nan
        if row['bull_regime']:
            return row['bull_volatility']
        else:
            return row['bear_volatility']
    
    df['regime_volatility'] = df.apply(regime_volatility, axis=1)
    
    # Take reciprocal for scaling (avoid division by zero)
    df['volatility_scaling'] = 1 / (df['regime_volatility'] + 1e-8)
    
    # Apply different scaling factors per regime
    def regime_scaling_factor(row):
        if pd.isna(row['volatility_scaling']):
            return np.nan
        if row['bull_regime']:
            return row['volatility_scaling'] * 1.0  # Normal scaling in bull
        else:
            return row['volatility_scaling'] * 1.5  # Enhanced scaling in bear
    
    df['final_scaling'] = df.apply(regime_scaling_factor, axis=1)
    
    # 5. Combine Components with Time Alignment
    # Multiply acceleration by volume divergence
    df['combined_signal'] = df['regime_scaled_acceleration'] * df['volume_signal']
    
    # Apply volatility scaling
    df['alpha_factor'] = df['combined_signal'] * df['final_scaling']
    
    # Ensure no future information leakage by shifting all calculations
    # All calculations use only data available at time t
    result = df['alpha_factor'].copy()
    
    return result

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Volume Confirmation alpha factor
    
    Parameters:
    df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
        Index should be datetime
    
    Returns:
    pandas Series with factor values indexed by date
    """
    
    # Multi-Horizon Momentum Construction
    df['M5'] = df['close'] / df['close'].shift(5) - 1
    df['M10'] = df['close'] / df['close'].shift(10) - 1
    df['M20'] = df['close'] / df['close'].shift(20) - 1
    
    # Volatility Regime Classification
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['rolling_volatility'] = df['daily_range'].rolling(window=10).std()
    
    # Regime classification
    df['volatility_regime'] = 'normal'
    df.loc[df['rolling_volatility'] > 0.02, 'volatility_regime'] = 'high'
    df.loc[df['rolling_volatility'] < 0.01, 'volatility_regime'] = 'low'
    
    # Volume Confirmation Signal
    def calculate_volume_percentile(volume_series):
        return volume_series.rank(pct=True) * 100
    
    df['volume_percentile'] = df['volume'].rolling(window=20).apply(
        lambda x: calculate_volume_percentile(pd.Series(x)).iloc[-1], 
        raw=False
    )
    df['volume_strength'] = (df['volume_percentile'] / 100) ** 0.5
    
    # Momentum Alignment Check
    df['momentum_alignment'] = 'mixed'
    df.loc[(df['M5'] > 0) & (df['M10'] > 0) & (df['M20'] > 0), 'momentum_alignment'] = 'all_positive'
    df.loc[(df['M5'] < 0) & (df['M10'] < 0) & (df['M20'] < 0), 'momentum_alignment'] = 'all_negative'
    
    # Regime-Adaptive Momentum Combination
    df['combined_momentum'] = np.nan
    
    # High volatility regime - focus on short-term
    high_vol_mask = df['volatility_regime'] == 'high'
    df.loc[high_vol_mask, 'combined_momentum'] = df.loc[high_vol_mask, 'M5']
    
    # Low volatility regime - focus on long-term
    low_vol_mask = df['volatility_regime'] == 'low'
    df.loc[low_vol_mask, 'combined_momentum'] = df.loc[low_vol_mask, 'M20']
    
    # Normal volatility regime - balanced approach
    normal_vol_mask = df['volatility_regime'] == 'normal'
    df.loc[normal_vol_mask, 'combined_momentum'] = (
        df.loc[normal_vol_mask, 'M5'] + 
        df.loc[normal_vol_mask, 'M10'] + 
        df.loc[normal_vol_mask, 'M20']
    ) / 3
    
    # Signal Enhancement
    df['enhanced_signal'] = np.nan
    
    # Aligned momentum cases with volume confirmation
    aligned_mask = df['momentum_alignment'].isin(['all_positive', 'all_negative'])
    df.loc[aligned_mask, 'enhanced_signal'] = (
        df.loc[aligned_mask, 'combined_momentum'] * 
        df.loc[aligned_mask, 'volume_strength']
    )
    
    # Mixed momentum cases with reduced weight
    mixed_mask = df['momentum_alignment'] == 'mixed'
    df.loc[mixed_mask, 'enhanced_signal'] = df.loc[mixed_mask, 'combined_momentum'] * 0.5
    
    # Regime Magnitude Adjustment
    df['alpha_factor'] = df['enhanced_signal']
    
    # High volatility: apply 0.8 multiplier
    df.loc[high_vol_mask, 'alpha_factor'] = df.loc[high_vol_mask, 'enhanced_signal'] * 0.8
    
    # Low volatility: apply 1.2 multiplier
    df.loc[low_vol_mask, 'alpha_factor'] = df.loc[low_vol_mask, 'enhanced_signal'] * 1.2
    
    # Normal volatility: no additional scaling (already handled)
    
    return df['alpha_factor']

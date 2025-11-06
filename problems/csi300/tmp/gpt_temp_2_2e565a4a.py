import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Volatility Asymmetry Identification
    df['positive_returns'] = np.where(df['returns'] > 0, df['returns'], np.nan)
    df['negative_returns'] = np.where(df['returns'] < 0, df['returns'], np.nan)
    
    df['upside_vol'] = df['positive_returns'].rolling(window=10, min_periods=5).std()
    df['downside_vol'] = df['negative_returns'].rolling(window=10, min_periods=5).std()
    df['vol_asymmetry_ratio'] = df['upside_vol'] / df['downside_vol']
    
    # Momentum Reversal Patterns
    # Short-Term Reversal Detection
    df['momentum_2d'] = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    df['momentum_extreme_rank'] = df['momentum_2d'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['reversal_prob_short'] = np.where(
        df['momentum_extreme_rank'] > 0.8, -1, 
        np.where(df['momentum_extreme_rank'] < 0.2, 1, 0)
    )
    
    # Medium-Term Mean Reversion
    df['ma_20'] = df['close'].rolling(window=20, min_periods=10).mean()
    df['momentum_8d'] = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    df['momentum_deviation'] = (df['close'] - df['ma_20']) / df['ma_20']
    
    # Persistence count
    df['return_sign'] = np.sign(df['returns'])
    df['sign_change'] = (df['return_sign'] != df['return_sign'].shift(1)).astype(int)
    df['persistence_count'] = df.groupby(df['sign_change'].cumsum()).cumcount() + 1
    
    df['mean_reversion_strength'] = -df['momentum_deviation'] * df['persistence_count']
    
    # Cross-Timeframe Reversal Confirmation
    df['reversal_confidence'] = (
        df['reversal_prob_short'] * np.sign(df['mean_reversion_strength']) * 
        (1 - 2 * abs(df['momentum_extreme_rank'] - 0.5))
    )
    
    # Volume-Based Reversal Validation
    df['volume_median_10d'] = df['volume'].rolling(window=10, min_periods=5).median()
    df['volume_spike'] = df['volume'] / df['volume_median_10d']
    df['volume_spike_magnitude'] = np.where(
        df['volume_spike'] > 1.5, df['volume_spike'] - 1.5, 0
    )
    
    # Volume-Price Divergence
    df['volume_change_5d'] = df['volume'].pct_change(5)
    df['price_change_5d'] = df['close'].pct_change(5)
    df['volume_price_corr'] = df['volume_change_5d'].rolling(window=10, min_periods=5).corr(df['price_change_5d'])
    df['volume_divergence'] = np.where(
        (df['volume_spike'] > 1.2) & (abs(df['returns']) < 0.01), 1, 0
    )
    
    # Asymmetric Volume Response
    df['volume_up_regime'] = np.where(df['vol_asymmetry_ratio'] > 1, df['volume_spike'], 0)
    df['volume_down_regime'] = np.where(df['vol_asymmetry_ratio'] < 1, df['volume_spike'], 0)
    df['volume_validation_score'] = (
        df['volume_spike_magnitude'] * df['volume_divergence'] * 
        (1 + df['volume_up_regime'] - df['volume_down_regime'])
    )
    
    # Opening Gap Reversal Analysis
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_abs'] = abs(df['overnight_gap'])
    df['gap_median_20d'] = df['gap_abs'].rolling(window=20, min_periods=10).median()
    df['significant_gap'] = np.where(df['gap_abs'] > df['gap_median_20d'], 1, 0)
    
    df['gap_filling_momentum'] = (df['close'] - df['open']) / (abs(df['open'] - df['close'].shift(1)) + 1e-8)
    df['gap_reversal_prob'] = (
        -np.sign(df['overnight_gap']) * df['significant_gap'] * 
        (1 - 2 * abs(df['gap_filling_momentum']))
    )
    
    # Intraday Price Compression
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_ma_5d'] = df['daily_range'].rolling(window=5, min_periods=3).mean()
    df['range_compression_ratio'] = df['daily_range'] / df['range_ma_5d']
    
    df['compression_count'] = df['range_compression_ratio'].rolling(window=5, min_periods=3).apply(
        lambda x: (x < 0.8).sum(), raw=False
    )
    
    df['compression_reversal_signal'] = (
        df['compression_count'] * df['reversal_confidence'] * 
        (1 / (df['range_compression_ratio'] + 0.1))
    )
    
    # Integrated Factor Generation
    # Core Reversal Component
    df['core_reversal'] = (
        df['reversal_confidence'] * 0.4 + 
        df['mean_reversion_strength'] * 0.3 +
        df['compression_reversal_signal'] * 0.3
    ) * (1 + (df['vol_asymmetry_ratio'] - 1))
    
    # Volume Validation Layer
    df['volume_enhanced_reversal'] = df['core_reversal'] * (1 + df['volume_validation_score'])
    
    # Gap Analysis Integration
    df['gap_enhanced_reversal'] = df['volume_enhanced_reversal'] + df['gap_reversal_prob'] * 0.2
    
    # Range Compression Enhancement
    df['compression_filter'] = np.where(df['range_compression_ratio'] < 0.8, 1.5, 1.0)
    df['final_reversal_factor'] = df['gap_enhanced_reversal'] * df['compression_filter']
    
    # Final volatility-momentum reversal factor
    factor = df['final_reversal_factor']
    
    return factor

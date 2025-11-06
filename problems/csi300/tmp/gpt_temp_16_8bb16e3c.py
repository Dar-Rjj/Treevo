import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Multi-Scale Momentum Confluence with Structural Break Detection
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic technical indicators
    df['returns'] = df['close'].pct_change()
    df['ATR_10'] = calculate_atr(df, window=10)
    df['ATR_5'] = calculate_atr(df, window=5)
    df['ATR_20'] = calculate_atr(df, window=20)
    df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    
    # Structural Break Identification
    df = calculate_structural_breaks(df)
    
    # Multi-Scale Momentum Confluence
    df = calculate_multi_scale_momentum(df)
    
    # Structural Break - Momentum Integration
    df = calculate_break_momentum_integration(df)
    
    # Dynamic Regime Context
    df = calculate_dynamic_regime(df)
    
    # Final composite factor
    factor = calculate_final_factor(df)
    
    return factor

def calculate_atr(df, window=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    return true_range.rolling(window=window).mean()

def calculate_structural_breaks(df):
    """Calculate structural break components"""
    
    # Price Level Breaks
    df['recent_high'] = df['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    df['recent_low'] = df['low'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    
    df['break_above_high'] = (df['close'] > df['recent_high']).astype(float)
    df['break_below_low'] = (df['close'] < df['recent_low']).astype(float)
    
    df['break_magnitude'] = np.where(
        df['break_above_high'] == 1,
        (df['close'] - df['recent_high']) / df['ATR_10'],
        np.where(
            df['break_below_low'] == 1,
            (df['recent_low'] - df['close']) / df['ATR_10'],
            0
        )
    )
    
    # Support/Resistance Tests
    df['support_level'] = df['low'].rolling(window=10, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    df['resistance_level'] = df['high'].rolling(window=10, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    
    df['bounce_from_support'] = (
        (np.abs(df['low'] - df['support_level']) / df['ATR_10'] < 0.3) & 
        (df['close'] > df['open'])
    ).astype(float)
    
    df['rejection_from_resistance'] = (
        (np.abs(df['high'] - df['resistance_level']) / df['ATR_10'] < 0.3) & 
        (df['close'] < df['open'])
    ).astype(float)
    
    # Gap Analysis
    df['gap_size'] = np.abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_fill_pct'] = np.abs(df['close'] - df['close'].shift(1)) / np.maximum(np.abs(df['open'] - df['close'].shift(1)), 1e-8)
    df['gap_direction_persistence'] = np.sign(df['open'] - df['close'].shift(1)) * np.sign(df['close'] - df['open'])
    
    # Volume Structure Breaks
    df['volume_median_5'] = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: np.median(x[:-1]) if len(x) > 1 else np.nan)
    df['volume_spike'] = (df['volume'] / df['volume_median_5'] > 2.0).astype(float)
    df['volume_contraction'] = (df['volume'] / df['volume_median_5'] < 0.5).astype(float)
    
    # Break Composite Scoring
    df['price_break_score'] = (
        df['break_magnitude'] + 
        df['bounce_from_support'] + 
        df['rejection_from_resistance'] +
        df['gap_size'] * df['gap_direction_persistence']
    )
    
    df['volume_break_score'] = (
        df['volume_spike'] * 2 - 
        df['volume_contraction'] +
        (df['volume'] / df['volume_median_5'])
    )
    
    df['structural_break_score'] = (
        0.6 * df['price_break_score'] + 
        0.4 * df['volume_break_score']
    )
    
    return df

def calculate_multi_scale_momentum(df):
    """Calculate multi-scale momentum confluence"""
    
    # Ultra-Short Momentum (1-2 days)
    df['gap_momentum'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['early_strength'] = (df['high'] - df['open']) / np.maximum(df['open'] - df['low'], 1e-8)
    df['opening_range_capture'] = (df['close'] - df['open']) / np.maximum(df['high'] - df['low'], 1e-8)
    
    df['ultra_short_momentum'] = (
        df['gap_momentum'] + 
        df['early_strength'].fillna(0) + 
        df['opening_range_capture']
    )
    
    # Short-Term Momentum (3-7 days)
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_acceleration'] = (
        (df['close'] - df['close'].shift(3)) / 
        np.maximum(df['close'].shift(3) - df['close'].shift(6), 1e-8)
    )
    
    # Calculate range expansion
    df['range_3d_avg'] = (df['high'] - df['low']).rolling(window=3).mean()
    df['range_expansion'] = (df['high'] - df['low']) / df['range_3d_avg']
    
    df['short_term_momentum'] = (
        df['momentum_3d'] + 
        df['momentum_acceleration'].fillna(0) + 
        df['range_expansion'].fillna(0)
    )
    
    # Medium-Term Momentum (8-21 days)
    df['trend_consistency'] = (
        (df['close'] - df['close'].shift(10)) / 
        df['close'].shift(10).rolling(window=10).apply(
            lambda x: np.sum(np.abs(np.diff(x))) if len(x) > 1 else np.nan
        )
    )
    
    df['trend_acceleration'] = (
        (df['close'] - df['close'].shift(5)) / 
        np.maximum(df['close'].shift(5) - df['close'].shift(10), 1e-8)
    )
    
    df['medium_term_momentum'] = (
        df['trend_consistency'].fillna(0) + 
        df['trend_acceleration'].fillna(0)
    )
    
    # Multi-Scale Confluence
    df['momentum_confluence'] = (
        0.3 * df['ultra_short_momentum'] +
        0.4 * df['short_term_momentum'] +
        0.3 * df['medium_term_momentum']
    )
    
    return df

def calculate_break_momentum_integration(df):
    """Integrate structural breaks with momentum"""
    
    # Break-confirmed momentum
    df['break_confirmed_momentum'] = (
        df['momentum_confluence'] * df['structural_break_score']
    )
    
    # Break-momentum alignment
    break_direction = np.sign(df['structural_break_score'])
    momentum_direction = np.sign(df['momentum_confluence'])
    df['break_momentum_alignment'] = (break_direction == momentum_direction).astype(float)
    
    # Composite break-momentum factor
    df['break_momentum_integration'] = (
        df['break_confirmed_momentum'] * 
        (1 + df['break_momentum_alignment'])
    )
    
    return df

def calculate_dynamic_regime(df):
    """Calculate dynamic regime context"""
    
    # Liquidity Regimes
    df['volume_quantile_20'] = df['volume'].rolling(window=20).apply(
        lambda x: np.percentile(x[:-1], 80) if len(x) > 1 else np.nan
    )
    
    df['high_liquidity'] = (
        (df['volume'] > df['volume_quantile_20']) & 
        ((df['high'] - df['low']) < df['ATR_10'])
    ).astype(float)
    
    df['low_liquidity'] = (
        (df['volume'] < df['volume_quantile_20'].rolling(window=20).apply(
            lambda x: np.percentile(x[:-1], 20) if len(x) > 1 else np.nan
        )) | 
        ((df['high'] - df['low']) > 1.5 * df['ATR_10'])
    ).astype(float)
    
    # Volatility Clustering
    df['high_vol_cluster'] = (df['ATR_5'] > 1.5 * df['ATR_20']).astype(float)
    df['low_vol_cluster'] = (df['ATR_5'] < 0.7 * df['ATR_20']).astype(float)
    
    # Regime-adaptive weighting
    df['regime_weight'] = np.where(
        df['high_liquidity'] == 1, 1.2,
        np.where(
            df['low_liquidity'] == 1, 0.8,
            np.where(
                df['high_vol_cluster'] == 1, 0.9,
                np.where(
                    df['low_vol_cluster'] == 1, 1.1,
                    1.0
                )
            )
        )
    )
    
    return df

def calculate_final_factor(df):
    """Calculate final composite factor"""
    
    # Apply regime-adaptive weighting
    df['adaptive_factor'] = df['break_momentum_integration'] * df['regime_weight']
    
    # Normalize the factor
    rolling_mean = df['adaptive_factor'].rolling(window=20, min_periods=10).mean()
    rolling_std = df['adaptive_factor'].rolling(window=20, min_periods=10).std()
    
    final_factor = (df['adaptive_factor'] - rolling_mean) / rolling_std
    
    return final_factor

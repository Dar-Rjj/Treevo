import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Price-Volume Convergence with Breakout Confirmation alpha factor
    """
    data = df.copy()
    
    # Dual-Timeframe Price-Volume Dynamics
    # Calculate daily price returns and volume changes
    data['price_return'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Calculate rolling correlations
    data['corr_5d'] = data['price_return'].rolling(window=5).corr(data['volume_change'])
    data['corr_20d'] = data['price_return'].rolling(window=20).corr(data['volume_change'])
    
    # Compute Correlation Momentum
    data['corr_mom_short'] = data['corr_5d'].diff(3)
    data['corr_mom_medium'] = data['corr_20d'].diff(5)
    
    # Volatility Regime Identification
    # Calculate True Range and Average True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_20d'] = data['true_range'].rolling(window=20).mean()
    
    # Classify volatility regime using rolling percentile
    data['volatility_regime'] = 0  # 0: low, 1: high
    data['atr_percentile'] = data['atr_20d'].rolling(window=60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data.loc[data['atr_percentile'] > 0.7, 'volatility_regime'] = 1
    
    # Momentum Acceleration with Breakout Detection
    data['mom_5d'] = data['close'].pct_change(5)
    data['mom_10d'] = data['close'].pct_change(10)
    data['mom_divergence'] = data['mom_5d'] - data['mom_10d']
    
    # Track divergence direction changes
    data['mom_div_sign'] = np.sign(data['mom_divergence'])
    data['mom_div_change'] = data['mom_div_sign'].diff().abs()
    
    # Detect Breakout Events
    data['rolling_20d_high'] = data['high'].rolling(window=20).max()
    data['rolling_20d_low'] = data['low'].rolling(window=20).min()
    data['breakout_high'] = (data['close'] > data['rolling_20d_high'].shift(1)).astype(int)
    data['breakout_low'] = (data['close'] < data['rolling_20d_low'].shift(1)).astype(int)
    data['breakout_strength'] = np.where(
        data['breakout_high'] == 1, 
        (data['close'] - data['rolling_20d_high'].shift(1)) / data['close'],
        np.where(
            data['breakout_low'] == 1,
            (data['rolling_20d_low'].shift(1) - data['close']) / data['close'],
            0
        )
    )
    
    # Amount-Enhanced Signal Construction
    data['amount_roc_5d'] = data['amount'].pct_change(5)
    data['amount_roc_10d'] = data['amount'].pct_change(10)
    data['amount_per_trade'] = data['amount'] / data['volume']
    data['amount_per_trade_roc'] = data['amount_per_trade'].pct_change(5)
    
    # Enhance correlation and momentum signals with amount data
    data['enhanced_corr_mom_short'] = data['corr_mom_short'] * data['amount_roc_5d']
    data['enhanced_corr_mom_medium'] = data['corr_mom_medium'] * data['amount_roc_10d']
    data['enhanced_mom_div'] = data['mom_divergence'] * data['amount_per_trade_roc']
    
    # Convergence-Breakout Pattern Detection
    # Calculate range efficiency
    data['daily_range'] = data['high'] - data['low']
    data['abs_price_change'] = abs(data['close'] - data['close'].shift(1))
    data['range_efficiency'] = data['abs_price_change'] / data['daily_range']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Signal persistence tracking
    data['corr_convergence'] = (data['corr_mom_short'] > 0) & (data['corr_mom_medium'] > 0)
    data['corr_convergence_streak'] = data['corr_convergence'].astype(int) * (data['corr_convergence'].groupby((~data['corr_convergence']).cumsum()).cumcount() + 1)
    
    # Breakout persistence
    data['breakout_persistence'] = data[['breakout_high', 'breakout_low']].max(axis=1)
    data['breakout_streak'] = data['breakout_persistence'].groupby((~data['breakout_persistence'].astype(bool)).cumsum()).cumcount()
    
    # Momentum acceleration consistency
    data['mom_accel_consistent'] = (data['mom_divergence'] > data['mom_divergence'].shift(1)).astype(int)
    data['mom_accel_streak'] = data['mom_accel_consistent'].groupby((~data['mom_accel_consistent'].astype(bool)).cumsum()).cumcount()
    
    # Generate Pattern Strength Score
    # High volatility regime patterns
    high_vol_pattern = (
        (data['enhanced_corr_mom_short'].abs() * 1.5) +
        (data['enhanced_corr_mom_medium'].abs() * 0.8) +
        (data['breakout_strength'].abs() * 2.0) +
        (data['enhanced_mom_div'].abs() * 1.2)
    )
    
    # Low volatility regime patterns
    low_vol_pattern = (
        (data['enhanced_corr_mom_short'] * 2.0) +
        (data['enhanced_corr_mom_medium'] * 1.5) +
        (data['breakout_strength'] * 1.0) +
        (data['amount_per_trade_roc'] * 1.8)
    )
    
    # Regime-specific pattern strength
    data['pattern_strength'] = np.where(
        data['volatility_regime'] == 1,
        high_vol_pattern,
        low_vol_pattern
    )
    
    # Range Efficiency and Persistence Filtering
    # Apply multi-dimensional filtering
    data['persistence_score'] = (
        data['corr_convergence_streak'] * 0.3 +
        data['breakout_streak'] * 0.4 +
        data['mom_accel_streak'] * 0.3
    )
    
    # Apply regime-specific persistence thresholds
    data['persistence_filter'] = np.where(
        data['volatility_regime'] == 1,
        np.where(data['persistence_score'] >= 2, 1, 0.5),
        np.where(data['persistence_score'] >= 3, 1, 0.3)
    )
    
    # Final Composite Alpha Factor
    # Multi-timeframe correlation momentum component
    corr_component = (
        data['enhanced_corr_mom_short'] * 0.4 +
        data['enhanced_corr_mom_medium'] * 0.6
    )
    
    # Regime-adaptive momentum acceleration
    mom_component = np.where(
        data['volatility_regime'] == 1,
        data['enhanced_mom_div'] * 1.2,
        data['enhanced_mom_div'] * 0.8
    )
    
    # Amount-enhanced breakout confirmation
    breakout_component = data['breakout_strength'] * data['amount_roc_5d'].abs()
    
    # Final composite factor with all components
    alpha_factor = (
        corr_component * 0.35 +
        mom_component * 0.25 +
        breakout_component * 0.20 +
        data['pattern_strength'] * 0.20
    )
    
    # Apply range efficiency and persistence filters
    alpha_factor = alpha_factor * data['range_efficiency'] * data['persistence_filter']
    
    # Final scaling and cleaning
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], 0).fillna(0)
    
    return alpha_factor

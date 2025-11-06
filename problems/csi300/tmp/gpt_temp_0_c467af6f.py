import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum signals (3-5 timeframes)
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_8 = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    mom_13 = (df['close'] - df['close'].shift(13)) / df['close'].shift(13)
    mom_21 = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
    
    # Volume trend momentum
    vol_trend_5 = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean() - 1
    vol_trend_10 = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=40).mean() - 1
    
    # Volatility-normalized momentum using ATR-based volatility
    atr = ((df['high'] - df['low']).rolling(window=5).mean() + 
           abs(df['high'] - df['close'].shift(1)).rolling(window=5).mean() + 
           abs(df['low'] - df['close'].shift(1)).rolling(window=5).mean()) / 3
    
    vol_norm_mom_3 = mom_3 / (atr / df['close'] + 1e-7)
    vol_norm_mom_8 = mom_8 / (atr / df['close'] + 1e-7)
    vol_norm_mom_13 = mom_13 / (atr / df['close'] + 1e-7)
    vol_norm_mom_21 = mom_21 / (atr / df['close'] + 1e-7)
    
    # Binary volume alignment across timeframes
    vol_align_3 = ((mom_3 * vol_trend_5) > 0).astype(float)
    vol_align_8 = ((mom_8 * vol_trend_5) > 0).astype(float)
    vol_align_13 = ((mom_13 * vol_trend_10) > 0).astype(float)
    vol_align_21 = ((mom_21 * vol_trend_10) > 0).astype(float)
    
    # Regime detection using volatility regime classification
    vol_regime = (atr / df['close']).rolling(window=30).apply(
        lambda x: 1.0 if x.iloc[-1] > x.quantile(0.7) else 
                  (0.5 if x.iloc[-1] > x.quantile(0.3) else 0.2)
    )
    
    # Multiplicative combination with regime-aware weights
    enhanced_3 = vol_norm_mom_3 * (1 + vol_regime * vol_align_3)
    enhanced_8 = vol_norm_mom_8 * (1 + vol_regime * vol_align_8)
    enhanced_13 = vol_norm_mom_13 * (1 + vol_regime * vol_align_13)
    enhanced_21 = vol_norm_mom_21 * (1 + vol_regime * vol_align_21)
    
    # Dynamic timeframe weighting based on recent volatility regime
    regime_weight_short = 0.6 if vol_regime.iloc[-1] > 0.7 else 0.4
    regime_weight_medium = 0.3 if vol_regime.iloc[-1] > 0.7 else 0.4
    regime_weight_long = 0.1 if vol_regime.iloc[-1] > 0.7 else 0.2
    
    # Final blended factor
    final_factor = (regime_weight_short * (enhanced_3 + enhanced_8) / 2 + 
                   regime_weight_medium * enhanced_13 + 
                   regime_weight_long * enhanced_21)
    
    return final_factor

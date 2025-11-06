import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum hierarchy with volume-pressure confirmation and volatility regime adaptation.
    
    Interpretation:
    - Triple timeframe momentum (intraday, overnight, weekly) using percentile ranks for cross-sectional comparison
    - Volume-pressure divergence detection using multiplicative combinations of volume ratios
    - Volatility regime classification based on range expansion and volume intensity
    - Regime-adaptive weighting scheme that emphasizes different timeframes based on market conditions
    - Multiplicative signal combinations enhance robustness and reduce noise
    - Positive values indicate strong momentum with volume confirmation across regimes
    - Negative values suggest momentum breakdown or divergence patterns
    """
    
    # Multi-timeframe momentum components with range normalization
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Percentile rank normalization for momentum components (20-day lookback)
    intraday_rank = intraday_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x.dropna()) >= 10 else 0.5, raw=False
    )
    overnight_rank = overnight_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x.dropna()) >= 10 else 0.5, raw=False
    )
    weekly_rank = weekly_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x.dropna()) >= 10 else 0.5, raw=False
    )
    
    # Volume-pressure divergence using multiplicative combinations
    volume_ratio_3d = df['volume'] / (df['volume'].rolling(window=3).mean() + 1e-7)
    volume_ratio_8d = df['volume'] / (df['volume'].rolling(window=8).mean() + 1e-7)
    volume_pressure = volume_ratio_3d * volume_ratio_8d
    
    amount_ratio_3d = df['amount'] / (df['amount'].rolling(window=3).mean() + 1e-7)
    amount_ratio_8d = df['amount'] / (df['amount'].rolling(window=8).mean() + 1e-7)
    amount_pressure = amount_ratio_3d * amount_ratio_8d
    
    # Volume divergence signal using multiplicative combination
    volume_divergence = volume_pressure * amount_pressure * np.sign(volume_pressure * amount_pressure)
    
    # Volatility regime classification
    daily_range = (df['high'] - df['low']) / (df['close'] + 1e-7)
    range_expansion = daily_range.rolling(window=5).mean() / (daily_range.rolling(window=15).mean() + 1e-7)
    volume_intensity = volume_pressure.rolling(window=3).mean()
    
    # Multi-regime classification
    regime_type = np.where(
        (range_expansion > 1.4) & (volume_intensity > 1.3), 'high_vol_expansion',
        np.where(
            (range_expansion < 0.7) & (volume_intensity < 0.8), 'low_vol_contraction',
            np.where(
                range_expansion > 1.2, 'moderate_expansion',
                np.where(
                    range_expansion < 0.8, 'moderate_contraction',
                    'normal_regime'
                )
            )
        )
    )
    
    # Regime-adaptive momentum acceleration using multiplicative combinations
    momentum_acceleration = (
        intraday_rank * overnight_rank * np.sign(intraday_momentum * overnight_momentum) +
        weekly_rank * intraday_rank * np.sign(weekly_momentum * intraday_momentum) +
        overnight_rank * weekly_rank * np.sign(overnight_momentum * weekly_momentum)
    )
    
    # Dynamic regime weights with multiplicative enhancement
    intraday_weight = np.where(
        regime_type == 'high_vol_expansion', 0.45 * volume_pressure,
        np.where(
            regime_type == 'low_vol_contraction', 0.2 * volume_pressure,
            np.where(
                regime_type == 'moderate_expansion', 0.35 * volume_pressure,
                np.where(
                    regime_type == 'moderate_contraction', 0.25 * volume_pressure,
                    0.3 * volume_pressure
                )
            )
        )
    )
    
    overnight_weight = np.where(
        regime_type == 'high_vol_expansion', 0.25 * amount_pressure,
        np.where(
            regime_type == 'low_vol_contraction', 0.35 * amount_pressure,
            np.where(
                regime_type == 'moderate_expansion', 0.3 * amount_pressure,
                np.where(
                    regime_type == 'moderate_contraction', 0.4 * amount_pressure,
                    0.32 * amount_pressure
                )
            )
        )
    )
    
    weekly_weight = np.where(
        regime_type == 'high_vol_expansion', 0.3,
        np.where(
            regime_type == 'low_vol_contraction', 0.45,
            np.where(
                regime_type == 'moderate_expansion', 0.35,
                np.where(
                    regime_type == 'moderate_contraction', 0.35,
                    0.38
                )
            )
        )
    )
    
    # Combined alpha factor with hierarchical structure and multiplicative enhancement
    alpha_factor = (
        intraday_weight * intraday_rank * np.sign(intraday_weight * intraday_rank) +
        overnight_weight * overnight_rank * np.sign(overnight_weight * overnight_rank) +
        weekly_weight * weekly_rank * np.sign(weekly_weight * weekly_rank) +
        momentum_acceleration * 0.2 * np.sign(momentum_acceleration) +
        volume_divergence * 0.15 * np.sign(volume_divergence)
    )
    
    return alpha_factor

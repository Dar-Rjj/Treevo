import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Regime Adaptation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Patterns
    # Short-term divergence (5-day price return vs 5-day volume change correlation)
    price_5d_return = data['close'].pct_change(5)
    volume_5d_change = data['volume'].pct_change(5)
    short_divergence = -price_5d_return.rolling(10).corr(volume_5d_change)
    
    # Medium-term divergence (20-day price momentum vs 20-day amount momentum)
    price_20d_momentum = data['close'] / data['close'].shift(20) - 1
    amount_20d_momentum = data['amount'] / data['amount'].shift(20) - 1
    medium_divergence = price_20d_momentum - amount_20d_momentum
    
    # Long-term divergence (60-day price trend vs 60-day volume trend slope difference)
    def calc_slope(series, window):
        x = np.arange(window)
        def rolling_slope(y):
            if len(y) == window and not np.isnan(y).any():
                return np.polyfit(x, y, 1)[0]
            return np.nan
        return series.rolling(window).apply(rolling_slope, raw=True)
    
    price_60d_slope = calc_slope(data['close'], 60)
    volume_60d_slope = calc_slope(data['volume'], 60)
    long_divergence = price_60d_slope - volume_60d_slope
    
    # Market Microstructure Signals
    # Order flow imbalance (daily buy volume - sell volume) / total volume
    # Estimate buy/sell volume using close-high-low relationship
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    price_pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    order_flow_imbalance = price_pressure * data['volume'] / data['volume'].rolling(20).mean()
    
    # Price impact efficiency (absolute return / dollar volume)
    daily_return = data['close'].pct_change().abs()
    dollar_volume = data['close'] * data['volume']
    price_impact_efficiency = -daily_return / (dollar_volume.rolling(10).mean() + 1e-8)
    
    # Volume clustering persistence (consecutive high/low volume days pattern)
    volume_zscore = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
    high_volume_days = (volume_zscore > 1).astype(int)
    low_volume_days = (volume_zscore < -1).astype(int)
    
    def consecutive_pattern(series, window=5):
        def count_consecutive(arr):
            if len(arr) < window:
                return 0
            current = arr[-1]
            count = 1
            for i in range(len(arr)-2, len(arr)-window-1, -1):
                if i >= 0 and arr[i] == current:
                    count += 1
                else:
                    break
            return count if count >= 2 else 0
        return series.rolling(window).apply(count_consecutive, raw=True)
    
    volume_clustering = consecutive_pattern(high_volume_days) - consecutive_pattern(low_volume_days)
    
    # Adaptive Regime Framework
    # Market state detection (rolling price range expansion/contraction)
    daily_range = (data['high'] - data['low']) / data['close']
    range_20d_avg = daily_range.rolling(20).mean()
    range_5d_avg = daily_range.rolling(5).mean()
    market_regime = (range_5d_avg / range_20d_avg - 1).fillna(0)
    
    # Dynamic factor weighting based on regime sensitivity
    # High volatility regimes favor short-term signals, low volatility favors long-term
    regime_weight_short = np.abs(market_regime)
    regime_weight_long = 1 - np.abs(market_regime)
    
    # Combine all components with regime-adaptive weighting
    divergence_strength = (
        regime_weight_short * short_divergence.fillna(0) +
        (regime_weight_short + regime_weight_long) / 2 * medium_divergence.fillna(0) +
        regime_weight_long * long_divergence.fillna(0)
    )
    
    microstructure_confirmation = (
        order_flow_imbalance.fillna(0) +
        price_impact_efficiency.fillna(0) +
        volume_clustering.fillna(0)
    ) / 3
    
    # Final factor: divergence strength × microstructure confirmation × regime sensitivity
    final_factor = (
        divergence_strength * 
        microstructure_confirmation * 
        (1 + np.abs(market_regime))
    )
    
    # Normalize and clean
    factor_series = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor_series = (factor_series - factor_series.rolling(60).mean()) / (factor_series.rolling(60).std() + 1e-8)
    
    return factor_series

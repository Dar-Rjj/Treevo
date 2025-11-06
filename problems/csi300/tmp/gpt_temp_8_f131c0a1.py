import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Price Efficiency with Liquidity Flow Divergence alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Efficiency Spectrum Analysis
    # Intraday Efficiency Metrics
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['overnight_gap'] = (data['open'] / data['close'].shift(1) - 1).fillna(0)
    
    # Gap closure efficiency
    gap_direction = np.sign(data['overnight_gap'])
    intraday_move = data['close'] - data['open']
    data['gap_closure_efficiency'] = np.where(
        gap_direction != 0,
        -gap_direction * intraday_move / (data['open'] * abs(data['overnight_gap'])).replace(0, np.nan),
        0
    )
    
    # Volatility-adjusted efficiency
    daily_return = (data['close'] / data['close'].shift(1) - 1).fillna(0)
    daily_range = (data['high'] - data['low']) / data['close'].shift(1).replace(0, np.nan)
    data['volatility_efficiency'] = abs(daily_return) / daily_range.replace(0, np.nan)
    
    # Efficiency persistence (consecutive days with high efficiency)
    high_efficiency = (data['range_utilization'] > 0.7).astype(int)
    data['efficiency_persistence'] = high_efficiency * (high_efficiency.groupby(high_efficiency.ne(high_efficiency.shift()).cumsum()).cumcount() + 1)
    
    # Multi-timeframe Efficiency Comparison
    data['efficiency_5d'] = data['range_utilization'].rolling(window=5, min_periods=3).mean()
    data['efficiency_20d'] = data['range_utilization'].rolling(window=20, min_periods=10).mean()
    data['efficiency_trend'] = data['efficiency_5d'] - data['efficiency_20d']
    
    # Efficiency regime classification
    data['efficiency_regime'] = np.select(
        [
            data['range_utilization'] > 0.7,
            data['range_utilization'] < 0.3
        ],
        [2, 0],  # 2=high, 1=normal, 0=low
        default=1
    )
    
    # Price Path Complexity
    daily_direction = np.sign(data['close'] - data['open'])
    data['whipsaw_intensity'] = (daily_direction != daily_direction.shift(1)).astype(int).rolling(window=5, min_periods=3).mean()
    
    # 5-day return consistency
    returns_5d = data['close'].pct_change(periods=5)
    returns_1d = data['close'].pct_change()
    data['trend_smoothness'] = abs(returns_5d) / (abs(returns_1d).rolling(window=5, min_periods=3).std().replace(0, np.nan))
    
    # Price discovery efficiency
    data['price_discovery'] = 1 - (abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low']).replace(0, np.nan))
    
    # Liquidity Flow Dynamics
    # Volume Distribution Analysis
    data['volume_percentile'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    
    # Volume clustering patterns
    high_volume = (data['volume'] > data['volume'].rolling(window=10, min_periods=5).mean()).astype(int)
    data['volume_persistence'] = high_volume * (high_volume.groupby(high_volume.ne(high_volume.shift()).cumsum()).cumcount() + 1)
    
    # Volume-price divergence
    price_movement = abs(data['close'].pct_change().fillna(0))
    data['volume_price_divergence'] = (data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()) - (price_movement / price_movement.rolling(window=10, min_periods=5).mean())
    
    # Trade Size Evolution
    data['trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['trade_size_ratio'] = data['trade_size'] / data['trade_size'].rolling(window=10, min_periods=5).median().replace(0, np.nan)
    
    # Large trade impact assessment
    median_trade_size = data['trade_size'].rolling(window=10, min_periods=5).median()
    large_trades = (data['trade_size'] > 2 * median_trade_size).astype(int)
    data['large_trade_frequency'] = large_trades.rolling(window=5, min_periods=3).mean()
    
    # Liquidity Regime Classification
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=10).mean()
    data['liquidity_regime'] = np.select(
        [
            data['volume'] > 1.5 * volume_20d_avg,
            data['volume'] < 0.7 * volume_20d_avg
        ],
        [2, 0],  # 2=high, 1=normal, 0=low
        default=1
    )
    
    # Efficiency-Liquidity Interaction Patterns
    # Regime-specific efficiency behavior
    high_liquidity_mask = data['liquidity_regime'] == 2
    low_liquidity_mask = data['liquidity_regime'] == 0
    
    data['regime_efficiency_score'] = np.select(
        [
            high_liquidity_mask & (data['efficiency_regime'] == 2),
            high_liquidity_mask & (data['efficiency_regime'] == 0),
            low_liquidity_mask & (data['efficiency_regime'] == 2),
            low_liquidity_mask & (data['efficiency_regime'] == 0)
        ],
        [1.0, -1.0, 0.5, -0.5],  # Strong trend, reversal, weak trend, consolidation
        default=0
    )
    
    # Trade size impact on efficiency
    data['trade_size_efficiency'] = np.where(
        data['trade_size_ratio'] > 1.5,
        data['range_utilization'] * np.sign(data['trade_size_ratio'] - 1),
        data['range_utilization'] * 0.5
    )
    
    # Multi-timeframe Signal Integration
    data['efficiency_momentum'] = data['range_utilization'] - data['range_utilization'].shift(3)
    data['liquidity_trend'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean() - 1
    
    # Long-term regime persistence
    efficiency_regime_persistence = (data['efficiency_regime'] == data['efficiency_regime'].shift(1)).astype(int)
    liquidity_regime_persistence = (data['liquidity_regime'] == data['liquidity_regime'].shift(1)).astype(int)
    data['regime_persistence'] = (efficiency_regime_persistence.rolling(window=20, min_periods=10).mean() + 
                                 liquidity_regime_persistence.rolling(window=20, min_periods=10).mean()) / 2
    
    # Divergence Detection System
    # Price-Liquidity Divergence
    efficiency_zscore = (data['range_utilization'] - data['range_utilization'].rolling(window=20, min_periods=10).mean()) / data['range_utilization'].rolling(window=20, min_periods=10).std().replace(0, np.nan)
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / data['volume'].rolling(window=20, min_periods=10).std().replace(0, np.nan)
    data['price_liquidity_divergence'] = efficiency_zscore - volume_zscore
    
    # Multi-timeframe Divergence
    short_term_efficiency = data['range_utilization'].rolling(window=5, min_periods=3).mean()
    long_term_efficiency = data['range_utilization'].rolling(window=20, min_periods=10).mean()
    data['efficiency_timeframe_divergence'] = short_term_efficiency - long_term_efficiency
    
    # Composite Alpha Generation
    # Core efficiency-liquidity score
    core_score = (data['range_utilization'] * 0.3 + 
                 data['volatility_efficiency'] * 0.2 + 
                 data['price_discovery'] * 0.2 + 
                 data['volume_percentile'] * 0.3)
    
    # Divergence adjustment
    divergence_adjustment = (data['price_liquidity_divergence'] * 0.4 + 
                           data['efficiency_timeframe_divergence'] * 0.3 + 
                           data['volume_price_divergence'] * 0.3)
    
    # Multi-timeframe alignment
    timeframe_alignment = (data['efficiency_momentum'] * 0.4 + 
                          data['liquidity_trend'] * 0.3 + 
                          data['regime_persistence'] * 0.3)
    
    # Regime-specific scaling
    regime_scaling = np.select(
        [
            (data['efficiency_regime'] == 2) & (data['liquidity_regime'] == 2),
            (data['efficiency_regime'] == 0) & (data['liquidity_regime'] == 0),
            (data['efficiency_regime'] != data['liquidity_regime'])
        ],
        [1.2, 0.8, 1.0],  # Boost aligned regimes, penalize misaligned
        default=1.0
    )
    
    # Final composite alpha
    alpha = (core_score * 0.5 + 
            divergence_adjustment * 0.3 + 
            timeframe_alignment * 0.2) * regime_scaling
    
    # Clean up and return
    alpha_series = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Price-Volume Acceleration
    # Short-term dynamics (3-day)
    data['short_price_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['short_volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['short_divergence'] = data['short_price_momentum'] - data['short_volume_momentum']
    
    # Medium-term dynamics (10-day)
    data['medium_price_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['medium_volume_momentum'] = data['volume'] / data['volume'].shift(10) - 1
    data['medium_divergence'] = data['medium_price_momentum'] - data['medium_volume_momentum']
    
    # Acceleration framework
    data['price_acceleration'] = data['short_price_momentum'] - data['medium_price_momentum']
    data['volume_acceleration'] = data['short_volume_momentum'] - data['medium_volume_momentum']
    data['divergence_acceleration'] = data['short_divergence'] - data['medium_divergence']
    
    # Volatility-Pressure Regime Classification
    # Volatility structure
    data['volatility_ratio'] = data['close'].rolling(5).std() / data['close'].shift(5).rolling(5).std()
    data['daily_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_volatility'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Pressure dynamics
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['cumulative_pressure'] = data['intraday_pressure'].rolling(3).sum()
    data['pressure_regime'] = data['cumulative_pressure'] > data['cumulative_pressure'].rolling(10).mean()
    
    # Regime classification
    data['high_vol_pressure'] = (data['volatility_ratio'] > 1) & data['pressure_regime']
    data['low_vol_pressure'] = (data['volatility_ratio'] <= 1) & (~data['pressure_regime'])
    data['mixed_regime'] = ~(data['high_vol_pressure'] | data['low_vol_pressure'])
    
    # Pattern & Reversal Detection
    # Reversal patterns
    data['failed_breakout'] = (data['high'] > data['high'].shift(1)) & (data['close'] < data['close'].shift(1))
    data['failed_breakout_count'] = data['failed_breakout'].rolling(5).sum()
    data['gap_recovery'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['momentum_exhaustion'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Volume validation
    data['volume_spikes'] = data['volume'] / data['volume'].shift(1).rolling(9).mean()
    
    # Calculate volume-pressure correlation
    def rolling_corr(x, window):
        return x['volume'].rolling(window).corr(x['intraday_pressure'])
    
    data['volume_pressure_correlation'] = data.groupby(level=0).apply(
        lambda x: rolling_corr(x, 5)
    ).values
    
    data['volume_efficiency'] = abs(data['close'] - data['open']) / data['volume']
    
    # Trend analysis
    data['trend_persistence'] = np.sign(data['close'] - data['close'].shift(1)).rolling(5).sum()
    data['mean_reversion_potential'] = abs(data['close'] - data['close'].shift(10)) / data['close'].rolling(10).std()
    data['price_impact_efficiency'] = abs(data['close'] - data['open']) / data['volume']
    
    # Regime-Adaptive Signal Enhancement
    # High volatility-pressure regime
    data['acceleration_momentum'] = data['price_acceleration'] * data['divergence_acceleration'] * data['volatility_ratio']
    data['volume_confirmed_reversal'] = data['failed_breakout_count'] * data['gap_recovery'] * data['volume_spikes']
    data['pressure_accumulation'] = data['cumulative_pressure'] * data['intraday_pressure'] * data['momentum_exhaustion']
    
    # Low volatility-pressure regime
    data['efficiency_weighted_divergence'] = data['medium_divergence'] * data['daily_range_efficiency'] * data['volume_efficiency']
    data['trend_confirmed_momentum'] = data['trend_persistence'] * data['price_acceleration'] * data['pressure_regime']
    data['volume_aligned_signals'] = data['volume_pressure_correlation'] * data['mean_reversion_potential']
    
    # Mixed regime
    data['range_breakout'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['liquidity_quality'] = data['amount'] / data['volume']
    data['failed_breakout_penalty'] = 1 - data['failed_breakout_count'] / 5
    
    # Composite Alpha Synthesis
    # Base factor
    base_factor = -data['price_acceleration'] * data['volume_acceleration'] * data['cumulative_pressure']
    
    # Pattern enhancement
    pattern_enhancement = data['failed_breakout_count'] * data['gap_recovery']
    
    # Momentum alignment
    momentum_alignment = data['intraday_pressure'] * data['trend_persistence']
    
    # Volume confirmation
    volume_confirmation = data['volume_spikes'] * data['volume_efficiency']
    
    # Efficiency adjustment
    efficiency_adjustment = data['daily_range_efficiency'] * data['price_impact_efficiency']
    
    # Regime application
    regime_factor = np.where(
        data['high_vol_pressure'],
        data['acceleration_momentum'] * data['volume_confirmed_reversal'] * data['pressure_accumulation'],
        np.where(
            data['low_vol_pressure'],
            data['efficiency_weighted_divergence'] * data['trend_confirmed_momentum'] * data['volume_aligned_signals'],
            data['range_breakout'] * data['liquidity_quality'] * data['failed_breakout_penalty']
        )
    )
    
    # Final composite alpha
    alpha = (
        base_factor * 
        pattern_enhancement * 
        momentum_alignment * 
        volume_confirmation * 
        efficiency_adjustment * 
        regime_factor
    )
    
    return alpha

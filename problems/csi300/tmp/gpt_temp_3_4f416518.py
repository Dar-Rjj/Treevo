import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency Analysis
    # Intraday Range Efficiency
    data['daily_efficiency'] = np.where(
        data['high'] != data['low'],
        abs(data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    
    # Efficiency Regime Classification
    conditions = [
        data['daily_efficiency'] > 0.7,
        (data['daily_efficiency'] >= 0.3) & (data['daily_efficiency'] <= 0.7),
        data['daily_efficiency'] < 0.3
    ]
    choices = [2, 1, 0]  # High:2, Normal:1, Low:0
    data['efficiency_regime'] = np.select(conditions, choices, default=1)
    
    # Efficiency Momentum Dynamics
    data['efficiency_3d_change'] = data['daily_efficiency'] - data['daily_efficiency'].shift(3)
    data['efficiency_acceleration'] = (data['efficiency_3d_change'] - 
                                     data['efficiency_3d_change'].shift(3))
    
    # Efficiency Persistence
    data['efficiency_direction'] = np.sign(data['efficiency_3d_change'])
    data['efficiency_persistence'] = (data['efficiency_direction'] == 
                                    data['efficiency_direction'].shift(1)).astype(int)
    
    # Volume Participation Divergence
    # Volume Concentration Pattern
    data['volume_hl_ratio'] = np.where(
        data['volume'] > 0,
        data['volume'] / data['volume'].shift(1),
        1
    )
    
    data['volume_skewness'] = np.where(
        abs(data['close'] - data['close'].shift(1)) > 0,
        (data['volume'] - data['volume'].shift(1)) / abs(data['close'] - data['close'].shift(1)),
        0
    )
    
    # Volume Efficiency Divergence
    data['volume_per_price_movement'] = np.where(
        abs(data['close'] - data['close'].shift(1)) > 0,
        data['volume'] / abs(data['close'] - data['close'].shift(1)),
        data['volume']
    )
    
    # 3-day Volume Efficiency Trend (slope approximation)
    data['volume_efficiency_trend'] = (
        data['volume_per_price_movement'] - data['volume_per_price_movement'].shift(3)
    ) / 3
    
    # Volume-Amount Confirmation
    data['implied_price'] = np.where(
        data['volume'] > 0,
        data['amount'] / data['volume'],
        data['close']
    )
    data['trading_intensity'] = abs(data['implied_price'] - data['close']) / data['close']
    
    # Volume-Amount Coherence Score
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['implied_direction'] = np.sign(data['implied_price'] - data['implied_price'].shift(1))
    data['volume_amount_coherence'] = (data['price_direction'] == data['implied_direction']).astype(int)
    
    # Price-Volume Divergence Detection
    # Multi-Timeframe Price Momentum
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['momentum_persistence'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_3d'].shift(1))
    
    # Efficiency-Volume Direction Divergence
    data['efficiency_momentum_sign'] = np.sign(data['efficiency_3d_change'])
    data['volume_efficiency_sign'] = np.sign(data['volume_efficiency_trend'])
    data['divergence_strength'] = (
        abs(data['efficiency_3d_change']) - abs(data['volume_efficiency_trend'])
    )
    
    # Regime-Volume Consistency
    data['regime_volume_consistency'] = (
        data['efficiency_momentum_sign'] == data['volume_efficiency_sign']
    ).astype(int)
    
    # Market Microstructure Integration
    # Intraday Pressure Assessment
    data['opening_pressure'] = np.where(
        data['high'] != data['low'],
        (data['open'] - data['close'].shift(1)) / (data['high'] - data['low']),
        0
    )
    data['closing_momentum'] = np.where(
        data['high'] != data['low'],
        (data['close'] - data['low']) / (data['high'] - data['low']),
        0.5
    )
    data['intraday_reversal'] = data['opening_pressure'] - data['closing_momentum']
    
    # Volume-Volatility Relationship
    data['volume_to_range_ratio'] = np.where(
        data['high'] != data['low'],
        data['volume'] / (data['high'] - data['low']),
        data['volume']
    )
    data['volatility_efficiency'] = np.where(
        data['high'] != data['low'],
        abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']),
        0
    )
    
    # Breakout Confirmation System
    data['breakout_signal'] = np.where(
        data['high'] != data['low'],
        (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low']),
        0
    )
    data['pressure_accumulation'] = (data['close'] - data['open']) * data['volume']
    data['cumulative_pressure_5d'] = data['pressure_accumulation'].rolling(window=5, min_periods=1).sum()
    
    # Composite Alpha Generation
    # Primary Factor
    data['primary_factor'] = (
        data['efficiency_3d_change'] * 
        data['divergence_strength'] * 
        data['intraday_reversal']
    )
    
    # Secondary Factor
    data['secondary_factor'] = (
        data['regime_volume_consistency'] * 
        data['breakout_signal'] * 
        data['volume_amount_coherence']
    )
    
    # Efficiency Context Adjustment
    efficiency_multiplier = np.select(
        [data['efficiency_regime'] == 2, data['efficiency_regime'] == 1, data['efficiency_regime'] == 0],
        [1.3, 1.0, 0.8],
        default=1.0
    )
    
    # Volume Strength Enhancement
    volume_percentile = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)) if len(x) > 5 else False
    )
    volume_multiplier = np.where(volume_percentile, 1.2, 1.0)
    
    # Microstructure Confirmation Factor
    data['microstructure_confirmation'] = (
        data['volatility_efficiency'] * 
        data['closing_momentum'] * 
        data['cumulative_pressure_5d']
    )
    
    # Final Alpha Calculation
    data['base_composite'] = data['primary_factor'] * data['secondary_factor']
    data['adjusted_composite'] = data['base_composite'] * efficiency_multiplier * volume_multiplier
    data['final_alpha'] = data['adjusted_composite'] * data['microstructure_confirmation']
    
    # Clean up and return
    alpha_series = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series

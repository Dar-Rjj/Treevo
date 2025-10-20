import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Efficiency Divergence with Volatility-Regime Adaptation alpha factor
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Multi-Timeframe Momentum-Efficiency Analysis
    # Calculate price momentum for different timeframes
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_21d'] = data['close'] / data['close'].shift(21) - 1
    
    # Calculate efficiency metrics (price change per unit volume)
    data['efficiency_3d'] = (data['close'] - data['close'].shift(3)) / data['volume'].rolling(3).mean()
    data['efficiency_8d'] = (data['close'] - data['close'].shift(8)) / data['volume'].rolling(8).mean()
    data['efficiency_21d'] = (data['close'] - data['close'].shift(21)) / data['volume'].rolling(21).mean()
    
    # Momentum-efficiency divergence patterns
    data['momentum_efficiency_div_short'] = data['momentum_3d'] - data['efficiency_3d'].rolling(5).mean()
    data['momentum_efficiency_div_medium'] = data['momentum_8d'] - data['efficiency_8d'].rolling(8).mean()
    data['momentum_efficiency_div_long'] = data['momentum_21d'] - data['efficiency_21d'].rolling(13).mean()
    
    # Volume-Amount Structural Analysis
    data['volume_momentum_5d'] = data['volume'] / data['volume'].rolling(5).mean() - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].rolling(10).mean() - 1
    
    # Volume concentration (assuming first 30 minutes = first 1/8 of trading day)
    # Using intraday high-low range as proxy for concentration
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['volume_concentration'] = data['intraday_range'] / data['volume'].rolling(5).std()
    
    # Volume persistence using autocorrelation
    data['volume_autocorr_5d'] = data['volume'].rolling(5).apply(lambda x: x.autocorr(), raw=False)
    
    # Amount concentration patterns
    data['amount_per_volume'] = data['amount'] / data['volume']
    data['amount_concentration'] = data['amount_per_volume'].rolling(5).std() / data['amount_per_volume'].rolling(20).std()
    
    # Volatility Regime Classification
    # Calculate ATR
    data['tr'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['atr_20d'] = data['tr'].rolling(20).mean()
    data['atr_60d'] = data['tr'].rolling(60).mean()
    data['volatility_ratio'] = data['atr_20d'] / data['atr_60d']
    
    # Volatility regime classification
    data['volatility_regime'] = 1  # Normal regime by default
    data.loc[data['volatility_ratio'] > 1.2, 'volatility_regime'] = 2  # High volatility
    data.loc[data['volatility_ratio'] < 0.8, 'volatility_regime'] = 0  # Low volatility
    
    # Volatility compression using Bollinger Band width
    data['bb_upper'] = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
    data['bb_lower'] = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['close'].rolling(20).mean()
    data['volatility_compression'] = data['bb_width'].rolling(10).mean() / data['bb_width']
    
    # Structural Break Enhancement
    # Volume-price relationship shifts
    data['volume_price_corr'] = data['volume'].rolling(10).corr(data['close'])
    data['volume_price_shift'] = data['volume_price_corr'] - data['volume_price_corr'].rolling(20).mean()
    
    # Market memory via return autocorrelation
    data['return'] = data['close'].pct_change()
    data['return_autocorr_1d'] = data['return'].rolling(5).apply(lambda x: x.autocorr(lag=1), raw=False)
    data['return_autocorr_3d'] = data['return'].rolling(10).apply(lambda x: x.autocorr(lag=3), raw=False)
    
    # Liquidity-Efficiency Context
    data['bid_ask_proxy'] = (data['high'] - data['low']) / data['close']
    data['price_impact'] = abs(data['return']) / data['volume']
    data['overnight_gap'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_efficiency'] = (data['high'] - data['low']) / data['overnight_gap'].replace(0, np.nan)
    
    # Pattern Convergence & Divergence Recognition
    data['momentum_alignment'] = (data['momentum_3d'] * data['momentum_8d'] * data['momentum_21d']).apply(np.sign)
    data['efficiency_trend'] = data['efficiency_3d'].rolling(5).mean() - data['efficiency_21d'].rolling(13).mean()
    
    # Composite Alpha Factor Generation with Regime Adaptation
    def calculate_regime_factor(row):
        if row['volatility_regime'] == 2:  # High Volatility
            momentum_div = 0.5 * row['momentum_efficiency_div_short']
            volume_conf = 0.3 * row['volume_concentration']
            vol_transition = 0.2 * row['volatility_ratio']
            return momentum_div + volume_conf + vol_transition
            
        elif row['volatility_regime'] == 0:  # Low Volatility
            volume_persist = 0.5 * row['volume_autocorr_5d']
            momentum_accel = 0.3 * (row['momentum_3d'] - row['momentum_21d'])
            efficiency_trend = 0.2 * row['efficiency_trend']
            return volume_persist + momentum_accel + efficiency_trend
            
        else:  # Normal Regime
            momentum_eff = 0.4 * (row['momentum_efficiency_div_medium'] + row['momentum_efficiency_div_long']) / 2
            volume_struct = 0.3 * (row['volume_momentum_5d'] + row['amount_concentration']) / 2
            vol_compression = 0.2 * row['volatility_compression']
            price_range_eff = 0.1 * row['intraday_efficiency']
            return momentum_eff + volume_struct + vol_compression + price_range_eff
    
    # Apply regime-adaptive calculation
    data['regime_factor'] = data.apply(calculate_regime_factor, axis=1)
    
    # Apply structural break enhancement
    data['structural_enhancement'] = data['volume_price_shift'] * data['return_autocorr_3d']
    
    # Incorporate liquidity-efficiency context
    data['liquidity_context'] = -data['bid_ask_proxy'] * data['price_impact']
    
    # Integrate pattern convergence signals
    data['pattern_convergence'] = data['momentum_alignment'] * data['efficiency_trend']
    
    # Final composite alpha factor
    data['alpha_factor'] = (
        data['regime_factor'] + 
        0.3 * data['structural_enhancement'] + 
        0.2 * data['liquidity_context'] + 
        0.15 * data['pattern_convergence']
    )
    
    # Normalize the final factor
    data['alpha_factor_normalized'] = (
        data['alpha_factor'] - data['alpha_factor'].rolling(63).mean()
    ) / data['alpha_factor'].rolling(63).std()
    
    return data['alpha_factor_normalized']

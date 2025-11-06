import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume-Range Efficiency Momentum with Adaptive Regimes
    """
    data = df.copy()
    
    # Volatility Regime Identification
    data['hl_range_10d'] = (data['high'] - data['low']).rolling(window=10).mean()
    data['hl_range_50d_avg'] = (data['high'] - data['low']).rolling(window=50).mean()
    data['vol_regime'] = (data['hl_range_10d'] / data['hl_range_50d_avg']) > 1.0
    
    # Regime-Adaptive Price Momentum
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['regime_adaptive_momentum'] = np.where(
        data['vol_regime'], 
        data['price_momentum_5d'], 
        data['price_momentum_20d']
    )
    
    # Multi-Scale Volume Momentum
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    
    # Range Efficiency Momentum
    data['daily_return'] = data['close'] / data['close'].shift(1) - 1
    data['daily_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['range_efficiency'] = np.abs(data['daily_return']) / data['daily_range']
    data['efficiency_5d'] = data['range_efficiency'].rolling(window=5).mean()
    data['efficiency_10d'] = data['range_efficiency'].rolling(window=10).mean()
    
    # Price-Volume Direction Divergence
    data['price_dir'] = np.sign(data['regime_adaptive_momentum'])
    data['volume_dir_5d'] = np.sign(data['volume_momentum_5d'])
    data['bullish_div'] = ((data['price_dir'] > 0) & (data['volume_dir_5d'] < 0)).astype(int)
    data['bearish_div'] = ((data['price_dir'] < 0) & (data['volume_dir_5d'] > 0)).astype(int)
    
    # Magnitude-Based Divergence
    data['price_strength'] = np.abs(data['regime_adaptive_momentum'])
    data['volume_strength'] = np.abs(data['volume_momentum_5d'])
    data['magnitude_div'] = (data['price_strength'] - data['volume_strength']) / (data['price_strength'] + data['volume_strength'])
    
    # Range Efficiency Divergence
    data['efficiency_div'] = (data['regime_adaptive_momentum'] - data['efficiency_5d']) / (np.abs(data['regime_adaptive_momentum']) + np.abs(data['efficiency_5d']))
    
    # Intraday Momentum Confirmation
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_strength_3d'] = data['intraday_strength'].rolling(window=3).mean()
    
    # Volume Persistence Integration
    data['volume_rank'] = data['volume'].rolling(window=20).rank(pct=True)
    data['volume_autocorr'] = data['volume'].rolling(window=20).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Price Efficiency Assessment
    data['return_autocorr'] = data['daily_return'].rolling(window=20).apply(lambda x: x.autocorr(lag=1), raw=False)
    data['price_inefficiency'] = np.abs(data['return_autocorr'])
    
    # Liquidity Conditions Analysis
    data['dollar_volume'] = data['close'] * data['volume']
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['transaction_size'] = data['amount'] / data['volume']
    data['transaction_trend'] = data['transaction_size'].rolling(window=5).mean() / data['transaction_size'].rolling(window=20).mean()
    
    # Efficiency-Liquidity Integration
    data['efficiency_liquidity'] = data['price_inefficiency'] * data['dollar_volume'].rolling(window=10).mean()
    
    # Opening Gap Mean Reversion
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_5d_avg'] = data['opening_gap'].rolling(window=5).mean()
    data['gap_std'] = data['opening_gap'].rolling(window=20).std()
    data['significant_gap'] = np.abs(data['opening_gap']) > (2 * data['gap_std'])
    data['gap_mean_reversion'] = -data['opening_gap'] * data['significant_gap']
    
    # Volume Context Assessment
    data['volume_vs_20d'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['gap_reversion_signal'] = data['gap_mean_reversion'] * data['volume_vs_20d']
    
    # Amount-Return Correlation Framework
    data['return_1d'] = data['close'] / data['close'].shift(1) - 1
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Calculate rolling correlation between amount and returns
    data['amount_return_corr'] = data['amount'].rolling(window=10).corr(data['return_1d'])
    data['amount_momentum_integration'] = data['amount_return_corr'] * np.abs(data['return_5d'])
    
    # Adaptive Alpha Factor Synthesis
    # Combine divergence signals
    data['divergence_score'] = (
        data['bullish_div'] - data['bearish_div'] + 
        data['magnitude_div'] + 
        data['efficiency_div']
    )
    
    # Multi-dimensional signal integration
    data['momentum_divergence'] = data['regime_adaptive_momentum'] * data['divergence_score']
    
    # Apply time-decay to recent signals
    decay_weights = np.array([0.6, 0.3, 0.1])  # Recent signals weighted more heavily
    data['momentum_divergence_smooth'] = (
        data['momentum_divergence'].rolling(window=3).apply(
            lambda x: np.dot(x, decay_weights[:len(x)]), raw=False
        )
    )
    
    # Scale by range efficiency for volatility adjustment
    data['volatility_adjusted'] = data['momentum_divergence_smooth'] / (data['efficiency_5d'] + 1e-8)
    
    # Incorporate intraday momentum confirmation
    data['intraday_confirmed'] = data['volatility_adjusted'] * data['intraday_strength_3d']
    
    # Liquidity-efficiency weighting
    data['liquidity_weight'] = 1 / (data['efficiency_liquidity'] + 1e-8)
    data['liquidity_adjusted'] = data['intraday_confirmed'] * data['liquidity_weight']
    
    # Volume momentum confirmation
    data['volume_confirmation'] = (
        data['volume_momentum_3d'] * 0.4 + 
        data['volume_momentum_5d'] * 0.4 + 
        data['volume_momentum_10d'] * 0.2
    )
    
    # Final factor synthesis
    data['alpha_factor'] = (
        data['liquidity_adjusted'] * 0.3 +
        data['volume_confirmation'] * 0.25 +
        data['gap_reversion_signal'] * 0.2 +
        data['amount_momentum_integration'] * 0.15 +
        (data['efficiency_5d'] - data['efficiency_10d']) * 0.1
    )
    
    # Apply volume persistence filtering
    data['volume_persistence_filter'] = data['volume_autocorr'] * data['volume_rank']
    data['final_alpha'] = data['alpha_factor'] * data['volume_persistence_filter']
    
    return data['final_alpha']

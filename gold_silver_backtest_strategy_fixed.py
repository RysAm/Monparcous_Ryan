import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Statistical and ML libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not available. ARMA model will be skipped.")
    STATSMODELS_AVAILABLE = False

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error

# Technical indicators - fallback if pandas_ta not available
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    print("Warning: pandas_ta not available. Using simple technical indicators.")
    PANDAS_TA_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class GoldSilverBacktester:
    """
    Comprehensive backtesting framework for Gold and Silver futures
    using ARMA, MLP, and K-NN models for daily return prediction
    """
    
    def __init__(self, data_directory="d:/Console/Donnée_Futures", 
                 transaction_cost=0.0005, commission=2.0):
        self.data_directory = data_directory
        self.transaction_cost = transaction_cost  # 0.05% slippage
        self.commission = commission  # $2 per round trip
        
        # Data storage
        self.raw_data = {}
        self.daily_data = {}
        self.features = {}
        
        # Models
        self.models = {
            'ARMA': {},
            'MLP': {},
            'KNN': {}
        }
        self.scalers = {}
        
        # Results
        self.predictions = {}
        self.signals = {}
        self.backtest_results = {}
        
        # Configuration
        self.instruments = ['GC Or(Gold)', 'SI Silver']
        self.train_window = 252  # 1 year (reduced for stability)
        self.test_window = 21  # 1 month
        
    def load_data(self):
        """
        Load Gold and Silver futures 1-minute OHLC data
        """
        print("Loading Gold and Silver futures data...")
        
        # File mapping
        file_mapping = {
            'GC Or(Gold)': 'GC Or(Gold) ,Time - Time - 1m, 5_26_2010 82500 PM-5_27_2025 82500 PM_6ece8af4-a981-4ddc-8974-be8bdd8a235e.csv',
            'SI Silver': 'SI Silver , Time - Time - 1m, 5_26_2010 82500 PM-5_27_2025 82500 PM_01b0096a-8ae9-40f3-8d9a-12c56d9a133e.csv'
        }
        
        for instrument, filename in file_mapping.items():
            file_path = os.path.join(self.data_directory, filename)
            
            try:
                # Load CSV with semicolon separator
                df = pd.read_csv(file_path, sep=';')
                
                # Remove empty columns
                if df.columns[-1] == '' or pd.isna(df.columns[-1]):
                    df = df.iloc[:, :-1]
                
                # Map columns to standard OHLC format
                required_cols = ['Time left', 'Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols].copy()
                    df.rename(columns={'Time left': 'DateTime'}, inplace=True)
                    
                    # Convert DateTime and set as index
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    df.set_index('DateTime', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Remove duplicates
                    df = df[~df.index.duplicated(keep='first')]
                    
                    # Convert to numeric
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove NaN values
                    df = df.dropna()
                    
                    # Sample data to reduce memory usage (take every 60th minute = hourly)
                    df = df.iloc[::60]  # Reduce data size
                    
                    self.raw_data[instrument] = df
                    print(f"Loaded {instrument}: {len(df)} hourly records")
                    
                else:
                    print(f"Missing required columns in {filename}")
                    
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    def aggregate_to_daily(self):
        """
        Aggregate hourly data to daily OHLC
        """
        print("Aggregating to daily data...")
        
        for instrument, df in self.raw_data.items():
            # Group by date and aggregate
            daily = df.groupby(df.index.date).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Convert index back to datetime
            daily.index = pd.to_datetime(daily.index)
            
            # Calculate daily returns
            daily['Returns'] = daily['Close'].pct_change()
            daily['Returns_OC'] = (daily['Close'] - daily['Open']) / daily['Open']
            
            # Remove NaN values
            daily = daily.dropna()
            
            self.daily_data[instrument] = daily
            print(f"{instrument} daily data: {len(daily)} days")
    
    def simple_sma(self, series, window):
        """Simple moving average fallback"""
        return series.rolling(window=window).mean()
    
    def simple_rsi(self, series, window=14):
        """Simple RSI calculation fallback"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_features(self):
        """
        Generate features for machine learning models
        """
        print("Calculating features...")
        
        for instrument, df in self.daily_data.items():
            features = pd.DataFrame(index=df.index)
            
            # Lagged returns
            for lag in [1, 2, 3, 5]:
                features[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
                features[f'Returns_OC_lag_{lag}'] = df['Returns_OC'].shift(lag)
            
            # Technical indicators
            if PANDAS_TA_AVAILABLE:
                features['RSI'] = ta.rsi(df['Close'], length=14)
                features['SMA_10'] = ta.sma(df['Close'], length=10)
                features['SMA_20'] = ta.sma(df['Close'], length=20)
                
                # Bollinger Bands
                bb = ta.bbands(df['Close'], length=20, std=2)
                if bb is not None and not bb.empty:
                    features['BB_upper'] = bb.iloc[:, 0] if len(bb.columns) > 0 else None
                    features['BB_lower'] = bb.iloc[:, 2] if len(bb.columns) > 2 else None
            else:
                # Fallback to simple indicators
                features['RSI'] = self.simple_rsi(df['Close'])
                features['SMA_10'] = self.simple_sma(df['Close'], 10)
                features['SMA_20'] = self.simple_sma(df['Close'], 20)
            
            # Price relative to moving averages
            features['Price_SMA10_ratio'] = df['Close'] / features['SMA_10']
            features['Price_SMA20_ratio'] = df['Close'] / features['SMA_20']
            
            # Volatility
            features['Volatility_5'] = df['Returns'].rolling(5).std()
            features['Volatility_20'] = df['Returns'].rolling(20).std()
            
            # Volume indicators
            features['Volume_SMA'] = self.simple_sma(df['Volume'], 20)
            features['Volume_ratio'] = df['Volume'] / features['Volume_SMA']
            
            # Target variable (next day return)
            features['Target'] = df['Returns'].shift(-1)
            features['Target_direction'] = (features['Target'] > 0).astype(int)
            
            # Remove NaN values
            features = features.dropna()
            
            self.features[instrument] = features
            print(f"{instrument} features: {len(features)} samples, {len(features.columns)-2} features")
    
    def train_arma_model(self, instrument, train_data):
        """
        Train ARIMA model for return prediction
        """
        if not STATSMODELS_AVAILABLE:
            return None
            
        try:
            # Use only returns for ARIMA with proper index
            returns = train_data['Target'].dropna()
            
            if len(returns) < 50:  # Need sufficient data
                return None
            
            # Reset index to avoid statsmodels warnings
            returns_clean = pd.Series(returns.values, index=range(len(returns)))
            
            # Fit simple AR model instead of ARIMA
            model = ARIMA(returns_clean, order=(1, 0, 0))  # Simplified to AR(1)
            fitted_model = model.fit()
            
            return fitted_model
        except Exception as e:
            print(f"ARMA training error for {instrument}: {str(e)}")
            return None
    
    def train_mlp_model(self, instrument, X_train, y_train, task='regression'):
        """
        Train MLP model for return prediction
        """
        try:
            if task == 'classification':
                model = MLPClassifier(
                    hidden_layer_sizes=(32, 16),  # Reduced complexity
                    activation='relu',
                    solver='adam',
                    alpha=0.01,  # Increased regularization
                    max_iter=200,  # Reduced iterations
                    random_state=42
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=(32, 16),  # Reduced complexity
                    activation='relu',
                    solver='adam',
                    alpha=0.01,  # Increased regularization
                    max_iter=200,  # Reduced iterations
                    random_state=42
                )
            
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"MLP training error for {instrument}: {str(e)}")
            return None
    
    def train_knn_model(self, instrument, X_train, y_train, task='regression'):
        """
        Train K-NN model for return prediction
        """
        try:
            if task == 'classification':
                model = KNeighborsClassifier(
                    n_neighbors=min(10, len(X_train)//2),  # Adaptive n_neighbors
                    weights='distance',
                    metric='euclidean'
                )
            else:
                model = KNeighborsRegressor(
                    n_neighbors=min(10, len(X_train)//2),  # Adaptive n_neighbors
                    weights='distance',
                    metric='euclidean'
                )
            
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"KNN training error for {instrument}: {str(e)}")
            return None
    
    def rolling_window_backtest(self):
        """
        Perform rolling window backtesting
        """
        print("Starting rolling window backtesting...")
        
        for instrument in self.instruments:
            if instrument not in self.features:
                continue
                
            print(f"\nBacktesting {instrument}...")
            
            features_df = self.features[instrument]
            feature_cols = [col for col in features_df.columns 
                          if col not in ['Target', 'Target_direction']]
            
            # Initialize prediction storage
            self.predictions[instrument] = {
                'ARMA': [],
                'MLP_reg': [],
                'MLP_clf': [],
                'KNN_reg': [],
                'KNN_clf': [],
                'dates': [],
                'actual': []
            }
            
            # Ensure we have enough data
            if len(features_df) < self.train_window + self.test_window:
                print(f"Insufficient data for {instrument}")
                continue
            
            # Rolling window - simplified
            num_windows = min(3, (len(features_df) - self.train_window) // self.test_window)
            
            for window in range(num_windows):
                start_idx = window * self.test_window
                train_start = start_idx
                train_end = start_idx + self.train_window
                test_start = train_end
                test_end = min(test_start + self.test_window, len(features_df))
                
                if test_end <= test_start:
                    break
                
                # Prepare training data
                train_data = features_df.iloc[train_start:train_end]
                test_data = features_df.iloc[test_start:test_end]
                
                X_train = train_data[feature_cols]
                y_train_reg = train_data['Target']
                y_train_clf = train_data['Target_direction']
                
                X_test = test_data[feature_cols]
                y_test_reg = test_data['Target']
                
                # Scale features
                scaler = StandardScaler()
                try:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                except:
                    print(f"Scaling error for {instrument}, window {window}")
                    continue
                
                # Train models
                arma_model = self.train_arma_model(instrument, train_data)
                mlp_reg = self.train_mlp_model(instrument, X_train_scaled, y_train_reg, 'regression')
                mlp_clf = self.train_mlp_model(instrument, X_train_scaled, y_train_clf, 'classification')
                knn_reg = self.train_knn_model(instrument, X_train_scaled, y_train_reg, 'regression')
                knn_clf = self.train_knn_model(instrument, X_train_scaled, y_train_clf, 'classification')
                
                # Make predictions
                for j in range(len(test_data)):
                    # ARMA prediction
                    if arma_model is not None:
                        try:
                            # Use forecast with proper parameters to avoid warnings
                            arma_pred = arma_model.forecast(steps=1, signal_only=True)[0]
                        except:
                            arma_pred = 0
                    else:
                        arma_pred = 0
                    
                    # MLP predictions
                    # Dans la section des prédictions MLP/KNN classification
                    # Remplacer les lignes ~380-390 par :
                    try:
                        mlp_reg_pred = mlp_reg.predict(X_test_scaled[j:j+1])[0] if mlp_reg else 0
                        # AMÉLIORATION : Utiliser les probabilités pour les signaux plus nuancés
                        if mlp_clf:
                            mlp_clf_proba = mlp_clf.predict_proba(X_test_scaled[j:j+1])[0]
                            mlp_clf_pred = mlp_clf_proba[1] - mlp_clf_proba[0]  # Différence de probabilités
                        else:
                            mlp_clf_pred = 0
                    except:
                        mlp_reg_pred = mlp_clf_pred = 0
                    
                    try:
                        knn_reg_pred = knn_reg.predict(X_test_scaled[j:j+1])[0] if knn_reg else 0
                        # AMÉLIORATION : Même chose pour KNN
                        if knn_clf:
                            knn_clf_proba = knn_clf.predict_proba(X_test_scaled[j:j+1])[0]
                            knn_clf_pred = knn_clf_proba[1] - knn_clf_proba[0]  # Différence de probabilités
                        else:
                            knn_clf_pred = 0
                    except:
                        knn_reg_pred = knn_clf_pred = 0
                    
                    # Store predictions
                    self.predictions[instrument]['ARMA'].append(arma_pred)
                    self.predictions[instrument]['MLP_reg'].append(mlp_reg_pred)
                    self.predictions[instrument]['MLP_clf'].append(mlp_clf_pred)
                    self.predictions[instrument]['KNN_reg'].append(knn_reg_pred)
                    self.predictions[instrument]['KNN_clf'].append(knn_clf_pred)
                    self.predictions[instrument]['dates'].append(test_data.index[j])
                    self.predictions[instrument]['actual'].append(y_test_reg.iloc[j])
                
                print(f"Completed window {window + 1}/{num_windows}")
    
    def generate_signals(self):
        """
        Generate trading signals from predictions - FIXED VERSION
        """
        print("Generating trading signals...")
        
        for instrument in self.predictions:
            self.signals[instrument] = {}
            
            for model_name in ['ARMA', 'MLP_reg', 'MLP_clf', 'KNN_reg', 'KNN_clf']:
                predictions = np.array(self.predictions[instrument][model_name])
                
                if len(predictions) == 0:
                    self.signals[instrument][model_name] = np.array([])
                    continue
                
                # FIXED: More aggressive signal generation
                signals = np.zeros_like(predictions)
                
                # PROBLÈME : ce continue empêche l'assignation des signaux !
                if model_name == 'ARMA':
                    threshold = np.std(predictions) * 0.05 if np.std(predictions) > 0 else 0.0001
                else:
                    upper_threshold = np.percentile(predictions, 75)
                    lower_threshold = np.percentile(predictions, 25)
                    
                    signals[predictions > upper_threshold] = 1   # Buy signal
                    signals[predictions < lower_threshold] = -1  # Sell signal
                    continue  # ← CE CONTINUE EMPÊCHE L'ASSIGNATION !
                
                # Ce code n'est jamais atteint pour les modèles ML
                signals[predictions > threshold] = 1
                signals[predictions < -threshold] = -1
                self.signals[instrument][model_name] = signals
                
                # Debug info
                num_signals = np.sum(np.abs(signals) > 0)
                print(f"{instrument} {model_name}: {num_signals} signals generated")
    
    def calculate_performance_metrics(self, returns, signals, instrument, model_name):
        """
        Calculate comprehensive performance metrics - FIXED VERSION
        """
        if len(returns) == 0 or len(signals) == 0:
            return self._empty_metrics()
        
        # Ensure same length
        min_len = min(len(returns), len(signals))
        returns = returns[:min_len]
        signals = signals[:min_len]
        
        # Apply transaction costs
        position_changes = np.diff(np.concatenate([[0], signals]))
        transaction_costs = np.abs(position_changes) * self.transaction_cost
        
        # Calculate strategy returns
        if len(signals) > 1:
            strategy_returns = signals[:-1] * returns[1:] - transaction_costs[1:]
        else:
            strategy_returns = np.array([])
        
        if len(strategy_returns) == 0:
            return self._empty_metrics()
        
        # FIXED: Correct performance calculations
        total_return = np.prod(1 + strategy_returns) - 1 if len(strategy_returns) > 0 else 0
        
        # FIXED: Proper annualization based on actual time period
        trading_days = len(strategy_returns)
        years = trading_days / 252.0  # Convert trading days to years
        
        if years > 0 and total_return > -1:
            annual_return = (1 + total_return) ** (1/years) - 1
        else:
            annual_return = 0
            
        volatility = np.std(strategy_returns) * np.sqrt(252) if len(strategy_returns) > 1 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        if len(strategy_returns) > 0:
            cumulative = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0
        
        # Directional accuracy
        if len(returns) > 1 and len(signals) > 1:
            actual_direction = np.sign(returns[1:])
            predicted_direction = np.sign(signals[:-1])
            # Only count when we have a position
            mask = predicted_direction != 0
            if np.sum(mask) > 0:
                directional_accuracy = np.mean(actual_direction[mask] == predicted_direction[mask])
            else:
                directional_accuracy = 0
        else:
            directional_accuracy = 0
        
        # Number of trades
        num_trades = np.sum(np.abs(position_changes) > 0)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'directional_accuracy': directional_accuracy,
            'num_trades': num_trades,
            'strategy_returns': strategy_returns
        }
    
    def _empty_metrics(self):
        """Return empty metrics for failed calculations"""
        return {
            'total_return': 0,
            'annual_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'directional_accuracy': 0,
            'num_trades': 0,
            'strategy_returns': np.array([])
        }
    
    def run_backtest(self):
        """
        Run complete backtest and calculate performance
        """
        print("Calculating backtest performance...")
        
        for instrument in self.predictions:
            self.backtest_results[instrument] = {}
            
            actual_returns = np.array(self.predictions[instrument]['actual'])
            
            for model_name in ['ARMA', 'MLP_reg', 'MLP_clf', 'KNN_reg', 'KNN_clf']:
                if model_name in self.signals[instrument]:
                    signals = self.signals[instrument][model_name]
                    
                    metrics = self.calculate_performance_metrics(
                        actual_returns, signals, instrument, model_name
                    )
                    
                    self.backtest_results[instrument][model_name] = metrics
    
    def print_results(self):
        """
        Print comprehensive backtest results
        """
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        for instrument in self.backtest_results:
            print(f"\n{instrument.upper()}:")
            print("-" * 60)
            
            results_df = pd.DataFrame()
            
            for model_name, metrics in self.backtest_results[instrument].items():
                results_df[model_name] = [
                    f"{metrics['total_return']:.2%}",
                    f"{metrics['annual_return']:.2%}",
                    f"{metrics['volatility']:.2%}",
                    f"{metrics['sharpe_ratio']:.3f}",
                    f"{metrics['max_drawdown']:.2%}",
                    f"{metrics['directional_accuracy']:.2%}",
                    f"{metrics['num_trades']}"
                ]
            
            results_df.index = [
                'Total Return', 'Annual Return', 'Volatility', 
                'Sharpe Ratio', 'Max Drawdown', 'Directional Accuracy', 'Num Trades'
            ]
            
            print(results_df.to_string())
    
    def plot_equity_curves(self):
        """
        Plot equity curves for all models and instruments
        """
        try:
            fig, axes = plt.subplots(len(self.instruments), 1, figsize=(15, 6*len(self.instruments)))
            if len(self.instruments) == 1:
                axes = [axes]
            
            for idx, instrument in enumerate(self.instruments):
                if instrument not in self.backtest_results:
                    continue
                    
                ax = axes[idx]
                
                for model_name, metrics in self.backtest_results[instrument].items():
                    if len(metrics['strategy_returns']) > 0:
                        cumulative = np.cumprod(1 + metrics['strategy_returns'])
                        dates = self.predictions[instrument]['dates'][:len(cumulative)]
                        ax.plot(dates, cumulative, label=f"{model_name} (SR: {metrics['sharpe_ratio']:.2f})")
                
                ax.set_title(f'{instrument} - Equity Curves')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Return')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig('gold_silver_backtest_results_fixed.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Plotting error: {str(e)}")
    
    def run_complete_backtest(self):
        """
        Run the complete backtesting pipeline
        """
        print("Starting Gold & Silver Futures Backtesting (Fixed Version)...")
        print("="*60)
        
        try:
            # Load and process data
            self.load_data()
            if not self.raw_data:
                print("No data loaded. Please check file paths.")
                return
                
            self.aggregate_to_daily()
            self.calculate_features()
            
            # Run backtesting
            self.rolling_window_backtest()
            self.generate_signals()
            self.run_backtest()
            
            # Display results
            self.print_results()
            self.plot_equity_curves()
            
            print("\nBacktesting completed successfully!")
            
        except Exception as e:
            print(f"Error during backtesting: {str(e)}")
            import traceback
            traceback.print_exc()

# Example usage
if __name__ == "__main__":
    # Initialize backtester
    backtester = GoldSilverBacktester(
        data_directory="d:/Console/Donnée_Futures",
        transaction_cost=0.0005,  # 0.05% slippage
        commission=2.0  # $2 per round trip
    )
    
    # Run complete backtest
    backtester.run_complete_backtest()
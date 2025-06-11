import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import warnings
import os
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

class MultiInstrumentORBStrategy:
    """
    Stratégie de trading Open Range Breakout multi-instruments
    
    Cette stratégie supporte plusieurs instruments, ferme les positions à la clôture
    du marché et calcule l'équité basée sur les contrats spécifiques.
    """
    
    def __init__(self, orb_period_minutes=30, stop_loss_ticks=10, take_profit_ticks=20):
        """
        Initialise la stratégie ORB multi-instruments
        
        Args:
            orb_period_minutes (int): Durée en minutes pour définir l'open range
            stop_loss_ticks (int): Nombre de ticks pour le stop loss
            take_profit_ticks (int): Nombre de ticks pour le take profit
        """
        self.orb_period_minutes = orb_period_minutes
        self.stop_loss_ticks = stop_loss_ticks
        self.take_profit_ticks = take_profit_ticks
        
        # Configuration des contrats et leurs spécifications
        self.contract_specs = {
            'GC': {'name': 'Gold', 'tick_size': 0.1, 'tick_value': 10, 'market_close': time(17, 0)},
            'SI': {'name': 'Silver', 'tick_size': 0.005, 'tick_value': 25, 'market_close': time(17, 0)},
            'ES': {'name': 'S&P 500', 'tick_size': 0.25, 'tick_value': 12.5, 'market_close': time(17, 0)},
            'NQ': {'name': 'Nasdaq', 'tick_size': 0.25, 'tick_value': 5, 'market_close': time(17, 0)},
            'YM': {'name': 'Dow Jones', 'tick_size': 1, 'tick_value': 5, 'market_close': time(17, 0)},
            'RTY': {'name': 'Russell 2000', 'tick_size': 0.1, 'tick_value': 5, 'market_close': time(17, 0)},
            'CL': {'name': 'Crude Oil', 'tick_size': 0.01, 'tick_value': 10, 'market_close': time(17, 0)},
            'NG': {'name': 'Natural Gas', 'tick_size': 0.001, 'tick_value': 10, 'market_close': time(17, 0)},
            'ZB': {'name': '30Y Treasury', 'tick_size': 0.03125, 'tick_value': 31.25, 'market_close': time(17, 0)},
            'ZN': {'name': '10Y Treasury', 'tick_size': 0.015625, 'tick_value': 15.625, 'market_close': time(17, 0)},
            'ZF': {'name': '5Y Treasury', 'tick_size': 0.0078125, 'tick_value': 7.8125, 'market_close': time(17, 0)},
            'ZT': {'name': '2Y Treasury', 'tick_size': 0.00390625, 'tick_value': 7.8125, 'market_close': time(17, 0)},
            '6E': {'name': 'Euro FX', 'tick_size': 0.00005, 'tick_value': 6.25, 'market_close': time(17, 0)},
            '6J': {'name': 'Japanese Yen', 'tick_size': 0.0000005, 'tick_value': 6.25, 'market_close': time(17, 0)},
            '6B': {'name': 'British Pound', 'tick_size': 0.0001, 'tick_value': 6.25, 'market_close': time(17, 0)},
            '6A': {'name': 'Australian Dollar', 'tick_size': 0.0001, 'tick_value': 10, 'market_close': time(17, 0)},
            '6C': {'name': 'Canadian Dollar', 'tick_size': 0.0001, 'tick_value': 10, 'market_close': time(17, 0)},
            '6S': {'name': 'Swiss Franc', 'tick_size': 0.0001, 'tick_value': 12.5, 'market_close': time(17, 0)},
            '6N': {'name': 'New Zealand Dollar', 'tick_size': 0.0001, 'tick_value': 10, 'market_close': time(17, 0)},
            'ZC': {'name': 'Corn', 'tick_size': 0.0025, 'tick_value': 12.5, 'market_close': time(14, 20)},
            'NKD': {'name': 'Nikkei 225', 'tick_size': 5, 'tick_value': 25, 'market_close': time(17, 0)}
        }
        
    def extract_symbol_from_filename(self, filename: str) -> str:
        """
        Extrait le symbole du contrat depuis le nom de fichier
        
        Args:
            filename (str): Nom du fichier
            
        Returns:
            str: Symbole du contrat
        """
        # Mapping des noms de fichiers vers les symboles
        symbol_mapping = {
            'GC Or': 'GC',
            'SI Silver': 'SI',
            'ES SP500': 'ES',
            'NQ Nasdaq': 'NQ',
            'YM Mini DOW': 'YM',
            'RTY Russel 2000': 'RTY',
            'CL Pétrole Brut': 'CL',
            'NG Natural Gas': 'NG',
            'ZB 30-Year': 'ZB',
            'ZN 10-Year': 'ZN',
            'ZF 5-Year': 'ZF',
            'ZT 2-Year': 'ZT',
            '6E Euro FX': '6E',
            '6J Yen Japonais': '6J',
            '6B Livre Sterling': '6B',
            '6A Dollar Australien': '6A',
            '6C Dollar Canadien': '6C',
            '6S Franc Suisse': '6S',
            '6N Dollar Néo-Zélandais': '6N',
            'ZC Maïs': 'ZC',
            'NKD Nikkei 225': 'NKD'
        }
        
        for key, symbol in symbol_mapping.items():
            if key in filename:
                return symbol
        
        # Si aucun mapping trouvé, essayer d'extraire les premiers caractères
        parts = filename.split(' ')
        if len(parts) > 0:
            return parts[0]
        
        return 'UNKNOWN'
    
    def load_data(self, file_path: str, years_back: int = 5) -> Tuple[pd.DataFrame, str]:
        """
        Charge les données depuis un fichier CSV pour une période spécifiée
        
        Args:
            file_path (str): Chemin vers le fichier de données
            years_back (int): Nombre d'années à charger depuis la fin
            
        Returns:
            tuple: (DataFrame des données, symbole du contrat)
        """
        try:
            # Extraction du symbole depuis le nom de fichier
            filename = os.path.basename(file_path)
            symbol = self.extract_symbol_from_filename(filename)
            
            # Lecture du fichier CSV avec séparateur point-virgule
            df = pd.read_csv(file_path, sep=';')
            
            # Renommage des colonnes si nécessaire
            if 'Time left' in df.columns:
                df = df.rename(columns={'Time left': 'datetime'})
            elif 'Time' in df.columns:
                df = df.rename(columns={'Time': 'datetime'})
            elif 'DateTime' in df.columns:
                df = df.rename(columns={'DateTime': 'datetime'})
            else:
                time_cols = [col for col in df.columns if 'time' in col.lower()]
                if time_cols:
                    df = df.rename(columns={time_cols[0]: 'datetime'})
                else:
                    raise ValueError("Aucune colonne de temps trouvée")
                    
            # Conversion de la colonne datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            
            # Vérification des colonnes nécessaires
            required_columns = ['Open', 'High', 'Low', 'Close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Colonne manquante: {col}")
                    
            # Tri par datetime
            df = df.sort_index()
            
            # Nettoyage des données
            df = df.dropna(subset=required_columns)
            
            # Filtrage pour les dernières années spécifiées
            end_date = df.index[-1]
            start_date = end_date - pd.DateOffset(years=years_back)
            df = df[df.index >= start_date]
            
            # Limitation pour optimiser les performances (échantillonnage)
            if len(df) > 200000:
                print(f"Limitation des données à 100000 lignes pour optimiser les performances")
                df = df.tail(200000)
            
            print(f"Données chargées pour {symbol}: {len(df)} lignes")
            print(f"Période: {df.index[0]} à {df.index[-1]}")
            
            return df, symbol
            
        except Exception as e:
            print(f"Erreur lors du chargement des données {file_path}: {e}")
            return None, None
    
    def is_market_hours(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Vérifie si le timestamp est pendant les heures de marché
        
        Args:
            timestamp (pd.Timestamp): Timestamp à vérifier
            symbol (str): Symbole du contrat
            
        Returns:
            bool: True si pendant les heures de marché
        """
        # Heures de marché simplifiées (peut être ajusté selon les besoins)
        market_open = time(9, 30)  # 9h30
        market_close = self.contract_specs.get(symbol, {}).get('market_close', time(17, 0))
        
        current_time = timestamp.time()
        return market_open <= current_time <= market_close
    
    def should_close_position(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Détermine si une position doit être fermée (proche de la clôture du marché)
        
        Args:
            timestamp (pd.Timestamp): Timestamp actuel
            symbol (str): Symbole du contrat
            
        Returns:
            bool: True si la position doit être fermée
        """
        market_close = self.contract_specs.get(symbol, {}).get('market_close', time(17, 0))
        current_time = timestamp.time()
        
        # Fermer les positions 30 minutes avant la clôture
        close_time = datetime.combine(timestamp.date(), market_close)
        close_time_minus_30 = (close_time - pd.Timedelta(minutes=30)).time()
        
        return current_time >= close_time_minus_30
    
    def identify_open_range(self, df: pd.DataFrame, date, symbol: str) -> Tuple:
        """
        Identifie l'open range pour une date donnée
        
        Args:
            df (pd.DataFrame): Données de prix
            date: Date pour laquelle calculer l'open range
            symbol (str): Symbole du contrat
            
        Returns:
            tuple: (high_range, low_range, start_time, end_time)
        """
        # Filtrer les données pour la date spécifiée
        day_data = df[df.index.date == date]
        
        if len(day_data) == 0:
            return None, None, None, None
            
        # Filtrer pour les heures de marché seulement
        market_data = day_data[day_data.index.to_series().apply(lambda x: self.is_market_hours(x, symbol))]
        
        if len(market_data) == 0:
            return None, None, None, None
            
        # Prendre les premières données de la journée de marché
        start_time = market_data.index[0]
        end_time = start_time + pd.Timedelta(minutes=self.orb_period_minutes)
        
        # Données de l'open range
        orb_data = market_data[(market_data.index >= start_time) & (market_data.index <= end_time)]
        
        if len(orb_data) == 0:
            return None, None, None, None
            
        high_range = float(orb_data['High'].max())
        low_range = float(orb_data['Low'].min())
        
        return high_range, low_range, start_time, end_time
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Génère les signaux de trading basés sur l'ORB
        
        Args:
            df (pd.DataFrame): Données de prix
            symbol (str): Symbole du contrat
            
        Returns:
            pd.DataFrame: DataFrame avec les signaux
        """
        df = df.copy()
        df['signal'] = 0
        df['orb_high'] = np.nan
        df['orb_low'] = np.nan
        df['position'] = 0
        
        # Obtenir toutes les dates uniques
        unique_dates = df.index.date
        unique_dates = sorted(list(set(unique_dates)))
        
        for date in unique_dates:
            # Identifier l'open range pour cette date
            high_range, low_range, start_time, end_time = self.identify_open_range(df, date, symbol)
            
            if high_range is None:
                continue
                
            # Marquer l'open range dans le DataFrame
            day_mask = df.index.date == date
            df.loc[day_mask, 'orb_high'] = high_range
            df.loc[day_mask, 'orb_low'] = low_range
            
            # Données après l'open range pour cette journée
            after_orb = df[(df.index.date == date) & (df.index > end_time)]
            
            position_taken = False
            
            for idx in after_orb.index:
                if position_taken:
                    break
                    
                # Vérifier si on est pendant les heures de marché
                if not self.is_market_hours(idx, symbol):
                    continue
                    
                current_price = df.at[idx, 'Close']
                
                # Conversion explicite en float si nécessaire
                if hasattr(current_price, 'iloc'):
                    current_price = float(current_price.iloc[0])
                else:
                    current_price = float(current_price)
                
                # Signal d'achat: prix dépasse le haut de l'open range
                if current_price > high_range:
                    df.loc[idx, 'signal'] = 1
                    position_taken = True
                    
                # Signal de vente: prix passe sous le bas de l'open range
                elif current_price < low_range:
                    df.loc[idx, 'signal'] = -1
                    position_taken = True
        
        return df
    
    def backtest_single_instrument(self, df: pd.DataFrame, symbol: str, initial_capital: float = 10000) -> Dict:
        """
        Effectue le backtest pour un seul instrument
        
        Args:
            df (pd.DataFrame): Données avec signaux
            symbol (str): Symbole du contrat
            initial_capital (float): Capital initial
            
        Returns:
            dict: Résultats du backtest
        """
        df_signals = self.generate_signals(df, symbol)
        
        capital = initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        equity_curve = []
        
        # Spécifications du contrat
        contract_spec = self.contract_specs.get(symbol, {
            'tick_size': 0.01, 'tick_value': 10, 'market_close': time(17, 0)
        })
        
        for idx in df_signals.index:
            current_price = df_signals.at[idx, 'Close']
            signal = df_signals.at[idx, 'signal']
            
            # Conversion explicite
            if hasattr(current_price, 'iloc'):
                current_price = float(current_price.iloc[0])
            else:
                current_price = float(current_price)
                
            if hasattr(signal, 'iloc'):
                signal = int(signal.iloc[0])
            else:
                signal = int(signal)
            
            # Fermeture forcée des positions à la clôture du marché
            if position != 0 and self.should_close_position(idx, symbol):
                if position > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                # Calcul du P&L en utilisant les spécifications du contrat
                price_change = abs(current_price - entry_price)
                ticks = price_change / contract_spec['tick_size']
                trade_pnl = ticks * contract_spec['tick_value'] * (1 if position > 0 else -1) * (1 if pnl_pct > 0 else -1)
                
                capital += trade_pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'pnl': trade_pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'market_close'
                })
                
                position = 0
            
            # Gestion des positions existantes
            elif position != 0:
                price_change = abs(current_price - entry_price)
                ticks_moved = price_change / contract_spec['tick_size']
                
                if position > 0:
                    # Position longue
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Vérifier SL/TP en ticks
                    if current_price <= entry_price - (self.stop_loss_ticks * contract_spec['tick_size']) or \
                       current_price >= entry_price + (self.take_profit_ticks * contract_spec['tick_size']):
                        
                        trade_pnl = ticks_moved * contract_spec['tick_value'] * (1 if current_price > entry_price else -1)
                        capital += trade_pnl
                        
                        exit_reason = 'stop_loss' if current_price <= entry_price - (self.stop_loss_ticks * contract_spec['tick_size']) else 'take_profit'
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': trade_pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_reason
                        })
                        
                        position = 0
                        
                elif position < 0:
                    # Position courte
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Vérifier SL/TP en ticks
                    if current_price >= entry_price + (self.stop_loss_ticks * contract_spec['tick_size']) or \
                       current_price <= entry_price - (self.take_profit_ticks * contract_spec['tick_size']):
                        
                        trade_pnl = ticks_moved * contract_spec['tick_value'] * (1 if current_price < entry_price else -1)
                        capital += trade_pnl
                        
                        exit_reason = 'stop_loss' if current_price >= entry_price + (self.stop_loss_ticks * contract_spec['tick_size']) else 'take_profit'
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': trade_pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_reason
                        })
                        
                        position = 0
            
            # Nouveaux signaux
            if position == 0 and signal != 0 and self.is_market_hours(idx, symbol):
                position = signal
                entry_price = current_price
                entry_time = idx
            
            # Calcul de l'equity actuelle
            current_equity = capital
            if position != 0:
                if position > 0:
                    unrealized_pnl_pct = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl_pct = (entry_price - current_price) / entry_price
                
                price_change = abs(current_price - entry_price)
                ticks = price_change / contract_spec['tick_size']
                unrealized_pnl = ticks * contract_spec['tick_value'] * (1 if unrealized_pnl_pct > 0 else -1)
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
        
        # Clôture de la dernière position si nécessaire
        if position != 0:
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                
            price_change = abs(current_price - entry_price)
            ticks = price_change / contract_spec['tick_size']
            trade_pnl = ticks * contract_spec['tick_value'] * (1 if pnl_pct > 0 else -1)
            
            capital += trade_pnl
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df_signals.index[-1],
                'entry_price': entry_price,
                'exit_price': current_price,
                'position': position,
                'pnl': trade_pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_data'
            })
        
        # Calcul des métriques de performance
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            total_return = (capital - initial_capital) / initial_capital
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            returns = trades_df['pnl_pct'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
            
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            total_return = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        results = {
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades_df,
            'equity_curve': equity_curve,
            'signals_df': df_signals
        }
        
        return results
    
    def backtest_multiple_instruments(self, data_folder: str, years_back: int = 1, initial_capital: float = 10000, test_mode: bool = False) -> Dict:
        """
        Effectue le backtest sur plusieurs instruments
        
        Args:
            data_folder (str): Dossier contenant les fichiers de données
            years_back (int): Nombre d'années à analyser
            initial_capital (float): Capital initial par instrument
            test_mode (bool): Si True, teste seulement quelques instruments principaux
            
        Returns:
            dict: Résultats consolidés du backtest
        """
        results = {}
        total_capital = 0
        total_trades = 0
        all_returns = []
        
        # Liste des fichiers CSV dans le dossier
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        # En mode test, limiter aux instruments principaux
        if test_mode:
            priority_instruments = ['GC Or', 'ES SP500', 'NQ Nasdaq', 'CL Pétrole', 'SI Silver']
            csv_files = [f for f in csv_files if any(instr in f for instr in priority_instruments)]
        
        print(f"Analyse de {len(csv_files)} instruments sur {years_back} ans...\n")
        
        for file in csv_files:
            file_path = os.path.join(data_folder, file)
            
            # Chargement des données
            df, symbol = self.load_data(file_path, years_back)
            
            if df is not None and symbol:
                print(f"\nBacktest pour {symbol} ({self.contract_specs.get(symbol, {}).get('name', symbol)})...")
                
                # Backtest pour cet instrument
                instrument_results = self.backtest_single_instrument(df, symbol, initial_capital)
                results[symbol] = instrument_results
                
                # Accumulation des statistiques globales
                total_capital += instrument_results['final_capital']
                total_trades += instrument_results['total_trades']
                
                if instrument_results['total_trades'] > 0:
                    all_returns.extend(instrument_results['trades']['pnl_pct'].tolist())
                
                # Affichage des résultats pour cet instrument
                print(f"  Rendement: {instrument_results['total_return']:.2%}")
                print(f"  Trades: {instrument_results['total_trades']}")
                print(f"  Taux de réussite: {instrument_results['win_rate']:.2%}")
                print(f"  Capital final: ${instrument_results['final_capital']:,.2f}")
        
        # Calcul des statistiques globales
        total_initial_capital = initial_capital * len(results)
        global_return = (total_capital - total_initial_capital) / total_initial_capital if total_initial_capital > 0 else 0
        global_sharpe = np.mean(all_returns) / np.std(all_returns) * np.sqrt(252) if len(all_returns) > 0 and np.std(all_returns) != 0 else 0
        
        summary = {
            'individual_results': results,
            'global_stats': {
                'total_initial_capital': total_initial_capital,
                'total_final_capital': total_capital,
                'global_return': global_return,
                'total_trades': total_trades,
                'global_sharpe': global_sharpe,
                'instruments_count': len(results)
            }
        }
        
        return summary
    
    def print_summary(self, results: Dict):
        """
        Affiche un résumé des résultats multi-instruments
        
        Args:
            results (dict): Résultats du backtest multi-instruments
        """
        print("\n" + "="*80)
        print("RÉSUMÉ DE LA STRATÉGIE OPEN RANGE BREAKOUT MULTI-INSTRUMENTS")
        print("="*80)
        
        global_stats = results['global_stats']
        individual_results = results['individual_results']
        
        print(f"Capital Initial Total: ${global_stats['total_initial_capital']:,.2f}")
        print(f"Capital Final Total: ${global_stats['total_final_capital']:,.2f}")
        print(f"Rendement Global: {global_stats['global_return']:.2%}")
        print(f"Nombre d'Instruments: {global_stats['instruments_count']}")
        print(f"Total des Trades: {global_stats['total_trades']}")
        print(f"Sharpe Ratio Global: {global_stats['global_sharpe']:.2f}")
        
        print("\nRésultats par Instrument:")
        print("-" * 80)
        
        # Tri des résultats par rendement
        sorted_results = sorted(individual_results.items(), 
                              key=lambda x: x[1]['total_return'], reverse=True)
        
        for symbol, result in sorted_results:
            contract_name = self.contract_specs.get(symbol, {}).get('name', symbol)
            print(f"{symbol:>6} ({contract_name:<20}): {result['total_return']:>8.2%} | "
                  f"Trades: {result['total_trades']:>3} | "
                  f"Win Rate: {result['win_rate']:>6.2%} | "
                  f"Capital: ${result['final_capital']:>10,.2f}")
        
        print("\nParamètres de la Stratégie:")
        print(f"- Période ORB: {self.orb_period_minutes} minutes")
        print(f"- Stop Loss: {self.stop_loss_ticks} ticks")
        print(f"- Take Profit: {self.take_profit_ticks} ticks")
        print("- Fermeture des positions: 30 min avant clôture du marché")
        print("="*80)
    
    def plot_results(self, results: Dict):
        """
        Affiche les graphiques des résultats multi-instruments
        
        Args:
            results (dict): Résultats du backtest multi-instruments
        """
        individual_results = results['individual_results']
        
        # Création des graphiques
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Résultats du Backtest Multi-Instruments - Stratégie Open Range Breakout', fontsize=16)
        
        # Graphique 1: Rendements par instrument
        symbols = list(individual_results.keys())
        returns = [individual_results[symbol]['total_return'] for symbol in symbols]
        
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax1.bar(symbols, returns, color=colors, alpha=0.7)
        ax1.set_title('Rendements par Instrument')
        ax1.set_ylabel('Rendement (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Graphique 2: Nombre de trades par instrument
        trades_count = [individual_results[symbol]['total_trades'] for symbol in symbols]
        ax2.bar(symbols, trades_count, alpha=0.7, color='blue')
        ax2.set_title('Nombre de Trades par Instrument')
        ax2.set_ylabel('Nombre de Trades')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Taux de réussite par instrument
        win_rates = [individual_results[symbol]['win_rate'] for symbol in symbols]
        ax3.bar(symbols, win_rates, alpha=0.7, color='orange')
        ax3.set_title('Taux de Réussite par Instrument')
        ax3.set_ylabel('Taux de Réussite (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%')
        ax3.legend()
        
        # Graphique 4: Distribution des rendements
        all_returns = []
        for symbol in symbols:
            if individual_results[symbol]['total_trades'] > 0:
                all_returns.extend(individual_results[symbol]['trades']['pnl_pct'].tolist())
        
        if all_returns:
            ax4.hist(all_returns, bins=30, alpha=0.7, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_title('Distribution des Rendements par Trade')
            ax4.set_xlabel('Rendement par Trade (%)')
            ax4.set_ylabel('Fréquence')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Fonction principale pour tester la stratégie multi-instruments
    """
    # Initialisation de la stratégie
    strategy = MultiInstrumentORBStrategy(
        orb_period_minutes=30,  # 30 minutes pour l'open range
        stop_loss_ticks=10,     # 10 ticks de stop loss
        take_profit_ticks=20    # 20 ticks de take profit
    )
    
    # Dossier contenant les données
    data_folder = "d:/Console/Donnée_Futures"
    
    print("Démarrage du backtest multi-instruments sur 1 an...")
    
    # Backtest sur plusieurs instruments (mode test pour les premiers essais)
    results = strategy.backtest_multiple_instruments(
        data_folder=data_folder,
        years_back=1,
        initial_capital=10000,
        test_mode=True  # Utiliser False pour analyser tous les instruments
    )
    
    # Affichage des résultats
    strategy.print_summary(results)
    
    # Affichage des graphiques
    strategy.plot_results(results)
    
    # Sauvegarde des résultats détaillés
    print("\nSauvegarde des résultats détaillés...")
    
    # Création d'un DataFrame consolidé des trades
    all_trades = []
    for symbol, result in results['individual_results'].items():
        if len(result['trades']) > 0:
            trades_df = result['trades'].copy()
            trades_df['symbol'] = symbol
            all_trades.append(trades_df)
    
    if all_trades:
        consolidated_trades = pd.concat(all_trades, ignore_index=True)
        consolidated_trades.to_csv('d:/Console/Backtest_stratégie/orb_multi_instrument_trades.csv', index=False)
        print("Trades sauvegardés dans: orb_multi_instrument_trades.csv")
        
        # Création d'un fichier texte détaillé pour analyse manuelle
        with open('d:/Console/Backtest_stratégie/analyse_trades_detaillee.txt', 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("ANALYSE DÉTAILLÉE DES TRADES - STRATÉGIE OPEN RANGE BREAKOUT\n")
            f.write("="*100 + "\n\n")
            
            # Résumé global
            f.write(f"Capital Initial Total: ${results['total_initial_capital']:,.2f}\n")
            f.write(f"Capital Final Total: ${results['total_final_capital']:,.2f}\n")
            f.write(f"Rendement Global: {results['total_return']:.2%}\n")
            f.write(f"Nombre Total de Trades: {results['total_trades']}\n")
            f.write(f"Sharpe Ratio Global: {results['sharpe_ratio']:.2f}\n\n")
            
            # Paramètres de la stratégie
            f.write("PARAMÈTRES DE LA STRATÉGIE:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Période ORB: {strategy.orb_period_minutes} minutes\n")
            f.write(f"Stop Loss: {strategy.stop_loss_ticks} ticks\n")
            f.write(f"Take Profit: {strategy.take_profit_ticks} ticks\n")
            f.write(f"Ratio Risque/Récompense: 1:{strategy.take_profit_ticks/strategy.stop_loss_ticks}\n\n")
            
            # Analyse par instrument
            for symbol, result in results['individual_results'].items():
                if len(result['trades']) > 0:
                    contract_spec = strategy.contract_specs.get(symbol, {})
                    f.write("="*80 + "\n")
                    f.write(f"INSTRUMENT: {symbol} ({contract_spec.get('name', 'Unknown')})\n")
                    f.write("="*80 + "\n")
                    f.write(f"Tick Size: {contract_spec.get('tick_size', 'N/A')}\n")
                    f.write(f"Tick Value: ${contract_spec.get('tick_value', 'N/A')}\n")
                    f.write(f"Capital Initial: ${result['initial_capital']:,.2f}\n")
                    f.write(f"Capital Final: ${result['final_capital']:,.2f}\n")
                    f.write(f"Rendement: {result['total_return']:.2%}\n")
                    f.write(f"Nombre de Trades: {result['total_trades']}\n")
                    f.write(f"Taux de Réussite: {result['win_rate']:.2%}\n")
                    f.write(f"Gain Moyen: ${result['avg_win']:.2f}\n")
                    f.write(f"Perte Moyenne: ${result['avg_loss']:.2f}\n")
                    f.write(f"Profit Factor: {result['profit_factor']:.2f}\n")
                    f.write(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}\n")
                    f.write(f"Max Drawdown: {result['max_drawdown']:.2%}\n\n")
                    
                    # Détail des trades
                    f.write("DÉTAIL DES TRADES:\n")
                    f.write("-" * 120 + "\n")
                    f.write(f"{'#':<4} {'Date Entrée':<20} {'Date Sortie':<20} {'Type':<6} {'Prix Entrée':<12} {'Prix Sortie':<12} {'P&L $':<12} {'P&L %':<10} {'Ticks':<8} {'Raison':<15}\n")
                    f.write("-" * 120 + "\n")
                    
                    trades_df = result['trades']
                    for idx, trade in trades_df.iterrows():
                        position_type = "LONG" if trade['position'] > 0 else "SHORT"
                        price_change = abs(trade['exit_price'] - trade['entry_price'])
                        ticks_moved = price_change / contract_spec.get('tick_size', 1)
                        
                        f.write(f"{idx+1:<4} {str(trade['entry_time']):<20} {str(trade['exit_time']):<20} {position_type:<6} ")
                        f.write(f"{trade['entry_price']:<12.4f} {trade['exit_price']:<12.4f} {trade['pnl']:<12.2f} ")
                        f.write(f"{trade['pnl_pct']:<10.2%} {ticks_moved:<8.1f} {trade['exit_reason']:<15}\n")
                    
                    # Statistiques des trades
                    winning_trades = trades_df[trades_df['pnl'] > 0]
                    losing_trades = trades_df[trades_df['pnl'] < 0]
                    
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("STATISTIQUES DÉTAILLÉES:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Trades Gagnants: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)\n")
                    f.write(f"Trades Perdants: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)\n")
                    
                    if len(winning_trades) > 0:
                        f.write(f"Gain Maximum: ${winning_trades['pnl'].max():.2f}\n")
                        f.write(f"Gain Minimum: ${winning_trades['pnl'].min():.2f}\n")
                    
                    if len(losing_trades) > 0:
                        f.write(f"Perte Maximum: ${losing_trades['pnl'].min():.2f}\n")
                        f.write(f"Perte Minimum: ${losing_trades['pnl'].max():.2f}\n")
                    
                    # Analyse des raisons de sortie
                    exit_reasons = trades_df['exit_reason'].value_counts()
                    f.write(f"\nRAISONS DE SORTIE:\n")
                    for reason, count in exit_reasons.items():
                        f.write(f"  {reason}: {count} trades ({count/len(trades_df)*100:.1f}%)\n")
                    
                    f.write("\n\n")
        
        print("Analyse détaillée sauvegardée dans: analyse_trades_detaillee.txt")
    
    print("\nBacktest terminé!")


if __name__ == "__main__":
    main()
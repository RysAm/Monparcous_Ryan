import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

class OpenRangeBreakoutStrategy:
    """
    Stratégie de trading basée sur l'Open Range Breakout (ORB)
    
    Cette stratégie identifie une plage de prix pendant les premières minutes
    de la session de trading et génère des signaux d'achat/vente quand le prix
    sort de cette plage.
    """
    
    def __init__(self, orb_period_minutes=30, stop_loss_pct=0.02, take_profit_pct=0.04):
        """
        Initialise la stratégie ORB
        
        Args:
            orb_period_minutes (int): Durée en minutes pour définir l'open range
            stop_loss_pct (float): Pourcentage de stop loss
            take_profit_pct (float): Pourcentage de take profit
        """
        self.orb_period_minutes = orb_period_minutes
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.positions = []
        self.trades = []
        self.current_position = None
        
    def load_data(self, file_path):
        """
        Charge les données depuis un fichier CSV
        
        Args:
            file_path (str): Chemin vers le fichier de données
            
        Returns:
            pd.DataFrame: Données formatées
        """
        try:
            # Lecture du fichier CSV avec séparateur point-virgule
            df = pd.read_csv(file_path, sep=';')
            
            # Affichage des colonnes pour debug
            print(f"Colonnes trouvées: {list(df.columns)}")
            
            # Renommage des colonnes si nécessaire
            if 'Time left' in df.columns:
                df = df.rename(columns={'Time left': 'datetime'})
            elif 'Time' in df.columns:
                df = df.rename(columns={'Time': 'datetime'})
            elif 'DateTime' in df.columns:
                df = df.rename(columns={'DateTime': 'datetime'})
            else:
                # Si aucune colonne de temps trouvée, chercher une colonne qui contient 'time'
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
            
            # Nettoyage des données - suppression des lignes avec des valeurs manquantes
            df = df.dropna(subset=required_columns)
            
            # Limitation des données pour les tests (dernières 50000 lignes)
            if len(df) > 50000:
                print(f"Limitation des données à 50000 lignes pour optimiser les performances")
                df = df.tail(50000)
            
            print(f"Données chargées: {len(df)} lignes")
            print(f"Période: {df.index[0]} à {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            print(f"Vérifiez que le fichier existe et que le format est correct.")
            return None
    
    def identify_open_range(self, df, date):
        """
        Identifie l'open range pour une date donnée
        
        Args:
            df (pd.DataFrame): Données de prix
            date (datetime.date): Date pour laquelle calculer l'open range
            
        Returns:
            tuple: (high_range, low_range, start_time, end_time)
        """
        # Filtrer les données pour la date spécifiée
        day_data = df[df.index.date == date]
        
        if len(day_data) == 0:
            return None, None, None, None
            
        # Prendre les premières données de la journée
        start_time = day_data.index[0]
        end_time = start_time + pd.Timedelta(minutes=self.orb_period_minutes)
        
        # Données de l'open range
        orb_data = day_data[(day_data.index >= start_time) & (day_data.index <= end_time)]
        
        if len(orb_data) == 0:
            return None, None, None, None
            
        high_range = float(orb_data['High'].max())
        low_range = float(orb_data['Low'].min())
        
        return high_range, low_range, start_time, end_time
    
    def generate_signals(self, df):
        """
        Génère les signaux de trading basés sur l'ORB
        
        Args:
            df (pd.DataFrame): Données de prix
            
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
            high_range, low_range, start_time, end_time = self.identify_open_range(df, date)
            
            if high_range is None:
                continue
                
            # Marquer l'open range dans le DataFrame
            day_mask = df.index.date == date
            df.loc[day_mask, 'orb_high'] = high_range
            df.loc[day_mask, 'orb_low'] = low_range
            
            # Données après l'open range pour cette journée
            after_orb = df[(df.index.date == date) & (df.index > end_time)]
            
            position_taken = False  # Pour éviter plusieurs signaux le même jour
            
            for idx in after_orb.index:
                if position_taken:
                    break
                    
                current_price = df.at[idx, 'Close']
                
                # Conversion explicite en float si nécessaire
                if hasattr(current_price, 'iloc'):
                    current_price = float(current_price.iloc[0])
                else:
                    current_price = float(current_price)
                
                # Signal d'achat: prix dépasse le haut de l'open range
                if current_price > high_range:
                    df.loc[idx, 'signal'] = 1  # Signal d'achat
                    position_taken = True
                    
                # Signal de vente: prix passe sous le bas de l'open range
                elif current_price < low_range:
                    df.loc[idx, 'signal'] = -1  # Signal de vente
                    position_taken = True
        
        return df
    
    def backtest(self, df, initial_capital=10000):
        """
        Effectue le backtest de la stratégie
        
        Args:
            df (pd.DataFrame): Données avec signaux
            initial_capital (float): Capital initial
            
        Returns:
            dict: Résultats du backtest
        """
        df_signals = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        equity_curve = []
        
        for idx in df_signals.index:
            current_price = df_signals.at[idx, 'Close']
            signal = df_signals.at[idx, 'signal']
            
            # Conversion explicite en float si nécessaire
            if hasattr(current_price, 'iloc'):
                current_price = float(current_price.iloc[0])
            else:
                current_price = float(current_price)
                
            if hasattr(signal, 'iloc'):
                signal = int(signal.iloc[0])
            else:
                signal = int(signal)
            
            # Gestion des positions existantes
            if position != 0:
                # Calcul du P&L actuel
                if position > 0:  # Position longue
                    pnl_pct = (current_price - entry_price) / entry_price
                    # Stop loss ou take profit
                    if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                        # Clôture de la position
                        trade_pnl = capital * position * pnl_pct
                        capital += trade_pnl
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': trade_pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'stop_loss' if pnl_pct <= -self.stop_loss_pct else 'take_profit'
                        })
                        
                        position = 0
                        
                elif position < 0:  # Position courte
                    pnl_pct = (entry_price - current_price) / entry_price
                    # Stop loss ou take profit
                    if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                        # Clôture de la position
                        trade_pnl = capital * abs(position) * pnl_pct
                        capital += trade_pnl
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': trade_pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'stop_loss' if pnl_pct <= -self.stop_loss_pct else 'take_profit'
                        })
                        
                        position = 0
            
            # Nouveaux signaux
            if position == 0 and signal != 0:
                position = signal * 0.95  # Utilise 95% du capital
                entry_price = current_price
                entry_time = idx
            
            # Calcul de l'equity actuelle
            current_equity = capital
            if position != 0:
                if position > 0:
                    unrealized_pnl = capital * position * (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl = capital * abs(position) * (entry_price - current_price) / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
        
        # Clôture de la dernière position si nécessaire
        if position != 0:
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                
            trade_pnl = capital * abs(position) * pnl_pct
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
            
            # Calcul du Sharpe ratio
            returns = trades_df['pnl_pct'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
            
            # Calcul du maximum drawdown
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
    
    def plot_results(self, results, symbol="Asset"):
        """
        Affiche les résultats du backtest
        
        Args:
            results (dict): Résultats du backtest
            symbol (str): Nom de l'actif
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Résultats du backtest - Stratégie Open Range Breakout - {symbol}', fontsize=16)
        
        df_signals = results['signals_df']
        trades_df = results['trades']
        
        # Graphique 1: Prix avec signaux et open range
        ax1.plot(df_signals.index, df_signals['Close'], label='Prix', alpha=0.7)
        ax1.plot(df_signals.index, df_signals['orb_high'], label='ORB High', color='red', alpha=0.5)
        ax1.plot(df_signals.index, df_signals['orb_low'], label='ORB Low', color='green', alpha=0.5)
        
        # Signaux d'achat
        buy_signals = df_signals[df_signals['signal'] == 1]
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Achat')
        
        # Signaux de vente
        sell_signals = df_signals[df_signals['signal'] == -1]
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Vente')
        
        ax1.set_title('Prix et Signaux de Trading')
        ax1.set_ylabel('Prix')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Courbe d'équité
        ax2.plot(df_signals.index, results['equity_curve'])
        ax2.set_title('Courbe d\'Équité')
        ax2.set_ylabel('Capital')
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Distribution des P&L
        if len(trades_df) > 0:
            ax3.hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Distribution des P&L par Trade')
            ax3.set_xlabel('P&L')
            ax3.set_ylabel('Fréquence')
            ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Métriques de performance
        metrics = [
            f"Rendement Total: {results['total_return']:.2%}",
            f"Nombre de Trades: {results['total_trades']}",
            f"Taux de Réussite: {results['win_rate']:.2%}",
            f"Profit Factor: {results['profit_factor']:.2f}",
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}",
            f"Max Drawdown: {results['max_drawdown']:.2%}"
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(metrics), transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Métriques de Performance')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results):
        """
        Affiche un résumé des résultats
        
        Args:
            results (dict): Résultats du backtest
        """
        print("\n" + "="*60)
        print("RÉSUMÉ DE LA STRATÉGIE OPEN RANGE BREAKOUT")
        print("="*60)
        print(f"Capital Initial: ${results['initial_capital']:,.2f}")
        print(f"Capital Final: ${results['final_capital']:,.2f}")
        print(f"Rendement Total: {results['total_return']:.2%}")
        print(f"Nombre Total de Trades: {results['total_trades']}")
        
        if results['total_trades'] > 0:
            print(f"Taux de Réussite: {results['win_rate']:.2%}")
            print(f"Gain Moyen: ${results['avg_win']:,.2f}")
            print(f"Perte Moyenne: ${results['avg_loss']:,.2f}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        
        print("\nParamètres de la Stratégie:")
        print(f"- Période ORB: {self.orb_period_minutes} minutes")
        print(f"- Stop Loss: {self.stop_loss_pct:.1%}")
        print(f"- Take Profit: {self.take_profit_pct:.1%}")
        print("="*60)


def main():
    """
    Fonction principale pour tester la stratégie
    """
    # Initialisation de la stratégie
    strategy = OpenRangeBreakoutStrategy(
        orb_period_minutes=30,  # 30 minutes pour l'open range
        stop_loss_pct=0.02,     # 2% de stop loss
        take_profit_pct=0.04    # 4% de take profit
    )
    
    # Exemple d'utilisation avec un fichier de données
    # Remplacez par le chemin vers votre fichier de données
    data_file = "d:/Console/Donnée_Futures/GC Or(Gold) ,Time - Time - 1m, 5_26_2010 82500 PM-5_27_2025 82500 PM_6ece8af4-a981-4ddc-8974-be8bdd8a235e.csv"
    
    print("Chargement des données...")
    df = strategy.load_data(data_file)
    
    if df is not None:
        print("\nExécution du backtest...")
        results = strategy.backtest(df, initial_capital=10000)
        
        # Affichage des résultats
        strategy.print_summary(results)
        
        # Affichage des graphiques
        strategy.plot_results(results, "Gold (GC)")
        
        # Affichage des premiers trades
        if len(results['trades']) > 0:
            print("\nPremiers trades:")
            print(results['trades'].head(10))
    else:
        print("Erreur lors du chargement des données.")


if __name__ == "__main__":
    main()
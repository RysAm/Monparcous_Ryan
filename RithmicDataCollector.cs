#region Using declarations
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript.Strategies;
using System;
using System.IO;
using System.ComponentModel.DataAnnotations;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RithmicDataCollector : Strategy
    {
        private StreamWriter dataWriter;
        private string dataDirectory = @"D:\Console\donne_historique";
        private string currentFileName;
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Collecteur de données Rithmic pour backtesting externe";
                Name = "RithmicDataCollector";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = false;
                // Propriétés de base uniquement
                BarsRequiredToTrade = 1;
            }
            else if (State == State.DataLoaded)
            {
                // Créer le répertoire s'il n'existe pas
                if (!Directory.Exists(dataDirectory))
                    Directory.CreateDirectory(dataDirectory);
                    
                // Nom du fichier basé sur l'instrument et la période
                string timeFrame = BarsPeriod.BarsPeriodType.ToString() + BarsPeriod.Value;
                currentFileName = $"{Instrument.FullName}_{timeFrame}_{DateTime.Now:yyyyMMdd}.csv";
                string filePath = Path.Combine(dataDirectory, currentFileName);
                
                // Créer ou ouvrir le fichier
                dataWriter = new StreamWriter(filePath, true);
                
                // Écrire l'en-tête si le fichier est nouveau
                if (new FileInfo(filePath).Length == 0)
                {
                    dataWriter.WriteLine("DateTime,Open,High,Low,Close,Volume,Timestamp");
                }
                
                Print($"Collecte de données démarrée : {filePath}");
            }
            else if (State == State.Terminated)
            {
                if (dataWriter != null)
                {
                    dataWriter.Close();
                    dataWriter.Dispose();
                    Print($"Collecte de données terminée : {currentFileName}");
                }
            }
        }
        
        protected override void OnBarUpdate()
        {
            if (dataWriter == null) return;
            
            try
            {
                // Format : DateTime,Open,High,Low,Close,Volume,Timestamp
                string line = $"{Time[0]:yyyy-MM-dd HH:mm:ss},{Open[0]},{High[0]},{Low[0]},{Close[0]},{Volume[0]},{Time[0].Ticks}";
                dataWriter.WriteLine(line);
                dataWriter.Flush(); // Assurer l'écriture immédiate
            }
            catch (Exception ex)
            {
                Print($"Erreur lors de l'écriture des données : {ex.Message}");
            }
        }
        
        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Répertoire de données", Description = "Répertoire où sauvegarder les données", Order = 1, GroupName = "Paramètres")]
        public string DataDirectory
        {
            get { return dataDirectory; }
            set { dataDirectory = value; }
        }
        #endregion
    }
}
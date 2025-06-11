#region Using declarations
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using NinjaTrader.NinjaScript.Indicators;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using WaveletSharp.Transforms;
using WaveletSharp.Filters;
using MathNet.Numerics.Statistics;  // <-- Ajouté
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class WaveletsBollingers : Strategy
    {
        private Series<double> denoisedClose;
        private Series<double> upperBandSeries;
        private Series<double> lowerBandSeries;

        private double[] priceBuffer;
        private int waveletDecompositionLevel = 3;
        private int barsSinceEntry;
        private DiscreteWaveletTransform _dwt;

        protected override void OnStateChange()
		{
		    try
		    {
		        if (State == State.SetDefaults)
		        {
		            Description = @"Stratégie Bollinger Bands avec réduction de bruit par ondelettes";
		            Name = "WaveletsBollingers";
		            Calculate = Calculate.OnBarClose;
		            NPeriod = 1350;
		            KCoefficient = 1.25;
		            StopLossPercent = 0.005;
		            IsOverlay = true;
		        }
		        else if (State == State.DataLoaded)
		        {
		            denoisedClose = new Series<double>(this);
		            upperBandSeries = new Series<double>(this);
		            lowerBandSeries = new Series<double>(this);
		            _dwt = new DiscreteWaveletTransform(new Daubechies4());
		
		            // Initialiser le buffer de prix maintenant
		            priceBuffer = new double[NPeriod];
		        }
		    }
		    catch (Exception ex)
		    {
		        Print("Erreur dans OnStateChange : " + ex.Message);
		        throw;
		    }
		}


        protected override void OnBarUpdate()
        {
            if (CurrentBar < NPeriod)
                return;

            // Initialisation du buffer de prix (lazy)
            if (priceBuffer == null || priceBuffer.Length != NPeriod)
                priceBuffer = new double[NPeriod];

            // Remplir le buffer
            for (int i = 0; i < NPeriod; i++)
                priceBuffer[i] = Close[i];

            // Normalisation
            double min = priceBuffer.Min();
            double max = priceBuffer.Max();
            double[] normalized = priceBuffer.Select(p => (p - min) / (max - min)).ToArray();

            // Décomposition DWT
            var decomposition = _dwt.Decompose(normalized, waveletDecompositionLevel);
            var approxs = decomposition.Approximations;
            var details = decomposition.Details;

            // Seuillage
            var thresholdedDetails = new List<double[]>();
            foreach (var detail in details)
            {
                double sigma = Statistics.StandardDeviation(detail);
                double threshold = sigma * Math.Sqrt(2 * Math.Log(detail.Length));
                thresholdedDetails.Add(detail.Select(c => Math.Abs(c) > threshold ? c : 0).ToArray());
            }

            // Reconstruction
            double[] denoisedNormalized = _dwt.Reconstruct(approxs, thresholdedDetails);
            double denoisedPrice = denoisedNormalized.Last() * (max - min) + min;
            denoisedClose[0] = denoisedPrice;

            // Moyenne et écart-type manuels
            int validCount = Math.Min(CurrentBar, NPeriod);
            List<double> recentDenoised = new List<double>();
            for (int i = 0; i < validCount; i++)
                recentDenoised.Add(denoisedClose[i]);

            double mean = recentDenoised.Average();
            double stdDev = Math.Sqrt(recentDenoised.Sum(x => Math.Pow(x - mean, 2)) / validCount);

            double upperBand = mean + KCoefficient * stdDev;
            double lowerBand = mean - KCoefficient * stdDev;

            upperBandSeries[0] = upperBand;
            lowerBandSeries[0] = lowerBand;

            // Signaux d'entrée
            if (CrossAbove(denoisedClose, upperBandSeries, 1))
                EnterLong();
            else if (CrossBelow(denoisedClose, lowerBandSeries, 1))
                EnterShort();

            // Gestion du stop-loss
            if (Position.MarketPosition != MarketPosition.Flat)
                barsSinceEntry++;
            else
                barsSinceEntry = 0;

            int stopLength = Math.Max(1, barsSinceEntry);
            if (Position.MarketPosition == MarketPosition.Long)
            {
                double maxPrice = MAX(High, stopLength)[0];
                SetStopLoss(CalculationMode.Price, maxPrice * (1 - StopLossPercent));
            }
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                double minPrice = MIN(Low, stopLength)[0];
                SetStopLoss(CalculationMode.Price, minPrice * (1 + StopLossPercent));
            }
        }

        #region Properties
        [NinjaScriptProperty]
        [Range(50, 5000)]
        public int NPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 3.0)]
        public double KCoefficient { get; set; }

        [NinjaScriptProperty]
        [Range(0.001, 0.1)]
        public double StopLossPercent { get; set; }
        #endregion
    }
}

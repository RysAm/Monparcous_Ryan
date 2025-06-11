#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Strategies
{
    public class StopHuntTester : Strategy
    {
        #region private variables
		private double 										upper;
		private double 										lower;
		private int 										dir=0;
		private int 										entryBar;
		private ATR											atr; 
		private FlexTrend									trend;			
		
		private int FromTime;
		private int ToTime;
		#endregion
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                                 = @"Stop Hunt Tester.";
                Name                                        = "Stop Hunt Tester";
                Calculate                                   = Calculate.OnBarClose;
                EntriesPerDirection                         = 1;
                EntryHandling                               = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy                = true;
                ExitOnSessionCloseSeconds                   = 30;
                IsFillLimitOnTouch                          = false;
                MaximumBarsLookBack                         = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution                         = OrderFillResolution.Standard;
                Slippage                                    = 0;
                StartBehavior                               = StartBehavior.WaitUntilFlat;
                TimeInForce                                 = TimeInForce.Gtc;
                TraceOrders                                 = false;
                RealtimeErrorHandling                       = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling                          = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade                         = 10;
                // Disable this property for performance gains in Strategy Analyzer optimizations
                
                // See the Help Guide for additional information
                IsInstantiatedOnEachOptimizationIteration   = true;

                // パラメータの初期値をセット
				StepSize=3;			

				ExitPeriod= 20;
				StartTime=7;
				Hours=5;
				}
			else if (State == State.Configure)
			{

		    }
			else if (State == State.Historical)
			{
				FromTime=StartTime*10000;
				ToTime=Math.Min(24,StartTime+Hours)*10000;
				atr		= ATR(50);
				trend		 = FlexTrend(1.2,10);
			}
        }

        protected override void OnBarUpdate()
        {
            //Add your custom indicator logic here.
           if(CurrentBars[0]<=6 || CurrentBars[0]<=3  )     return;
				//---
				double trend0=trend[0];
				double atr0=atr[0];
				//--- Exit Position
	            if(MarketPosition.Long == Position.MarketPosition	&& CurrentBars[0] - entryBar >= ExitPeriod)
				{
					ExitLong();
				}
	            if(MarketPosition.Short== Position.MarketPosition && CurrentBars[0] - entryBar >= ExitPeriod)
				{
					ExitShort();
				}
				//---
				
				if( trend0>3 )
				{
					if(dir==0|| dir<=-1)
					{
						upper=Close[0];
						dir= 1;
					}
					if(upper<Close[0])upper=Close[0];					
					if(trend0>15) dir=2;
				}
				
				if( trend0<-3)
				{
					if(dir==0||dir>=1)
					{
						lower=Close[0];				
						dir= -1;
					}
					if(lower>Close[0])lower=Close[0];					
					if(trend0<-15) dir=-2;
				}
				
				
				if(MarketPosition.Flat == Position.MarketPosition && checktime(ToTime(Time[0]))  )                        
		        {
					double stop=upper-atr0*StepSize; 
					if(dir==1 && trend0>5 && stop>Close[0] &&  Close[0]<Close[1])
					{
						entryBar=CurrentBars[0];	
						EnterLong(10000);
					}
				}
	        	if(MarketPosition.Flat == Position.MarketPosition && checktime(ToTime(Time[0])) )        
	            {
					double stop=lower+atr0*StepSize; 
					if(dir==-1&& trend0<-5 && stop < Close[0] &&  Close[0]>Close[1])
					{
						entryBar=CurrentBars[0];	
						EnterShort(10000);
					}
				}
            //---           
        }
		bool checktime(int now)
		{
			if((FromTime<=ToTime) && (now>=FromTime && now<ToTime))     return true;
	    	    
			if((FromTime>ToTime) && (now>=FromTime || now<ToTime))	return true;
        
			return false;
		}
        #region Properties

 

		
        [Range(0, int.MaxValue), NinjaScriptProperty]
        [Display(ResourceType = typeof(Custom.Resource), Name = "StopSize",
                                GroupName = "NinjaScriptParameters", Order = 0)]
        public double StepSize
        { get; set; }


        // period
        [Range(1, int.MaxValue), NinjaScriptProperty]
        [Display(ResourceType = typeof(Custom.Resource), Name = "Exit Period",
                                GroupName = "NinjaScriptParameters", Order = 1)]
        public int ExitPeriod
        { get; set; }

        [Range(0,24), NinjaScriptProperty]
        [Display(ResourceType = typeof(Custom.Resource), Name = "Start Time",
                                GroupName = "NinjaScriptParameters", Order = 2)]
        public int StartTime
        { get; set; }

        [Range(1, 24), NinjaScriptProperty]
        [Display(ResourceType = typeof(Custom.Resource), Name = "hours",
                                GroupName = "NinjaScriptParameters", Order = 3)]
        public int Hours
        { get; set; }


        #endregion
    }
}

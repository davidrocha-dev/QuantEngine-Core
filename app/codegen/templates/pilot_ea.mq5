//+------------------------------------------------------------------+
//| QuantEngine Pilot - REQ-PORT-4                                    |
//|                                                                   |
//| Records closed deals so QuantEngine can compare a deployed        |
//| strategy against the backtest that approved it. It never trades:  |
//| it only reads this terminal's own history and appends it to a CSV |
//| in the shared Files folder, where the engine picks it up.         |
//|                                                                   |
//| Attach it to any one chart. One instance per terminal is enough - |
//| it reports every QuantEngine EA running there, not just the       |
//| symbol it sits on.                                                |
//+------------------------------------------------------------------+
#property copyright "QuantEngine"
#property version   "1.00"
#property strict

input int    PollSeconds   = 30;                       // how often to check for new deals
input string OutputFile    = "quantengine_live.csv";   // written to the COMMON Files folder
input int    LookbackDays  = 30;                       // how far back to scan on first run

// Only deals from QuantEngine EAs are recorded. Generated strategy EAs use
// magic numbers in this band (see app/codegen/mql5.py `_magic_number`), so
// a hand-placed trade or another EA's position is never reported as if it
// were a strategy's result.
#define QE_MAGIC_MIN 100000000
#define QE_MAGIC_MAX 999999999

long LastReportedDeal = 0;

int OnInit()
{
   EventSetTimer(MathMax(5, PollSeconds));
   WriteHeaderIfMissing();
   ScanHistory();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

void OnTimer() { ScanHistory(); }

void WriteHeaderIfMissing()
{
   if(FileIsExist(OutputFile, FILE_COMMON)) return;

   int handle = FileOpen(OutputFile, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE) return;
   FileWrite(handle, "deal_ticket", "closed_at", "magic", "symbol", "volume", "profit", "entry_price", "exit_price");
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Appends every closed QuantEngine deal newer than the last one we  |
//| reported. Appending (never rewriting) means a restart cannot lose |
//| history, and the engine deduplicates on deal_ticket anyway.       |
//+------------------------------------------------------------------+
void ScanHistory()
{
   datetime from = TimeCurrent() - (datetime)LookbackDays * 86400;
   if(!HistorySelect(from, TimeCurrent())) return;

   int handle = FileOpen(OutputFile, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE) return;
   FileSeek(handle, 0, SEEK_END);

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if((long)ticket <= LastReportedDeal) continue;

      long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      if(magic < QE_MAGIC_MIN || magic > QE_MAGIC_MAX) continue;           // not ours
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;  // closes only

      FileWrite(
         handle,
         (string)ticket,
         TimeToString((datetime)HistoryDealGetInteger(ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
         (string)magic,
         HistoryDealGetString(ticket, DEAL_SYMBOL),
         DoubleToString(HistoryDealGetDouble(ticket, DEAL_VOLUME), 2),
         DoubleToString(HistoryDealGetDouble(ticket, DEAL_PROFIT), 2),
         DoubleToString(0.0, 5),                                           // entry price: see note below
         DoubleToString(HistoryDealGetDouble(ticket, DEAL_PRICE), 5)
      );

      if((long)ticket > LastReportedDeal) LastReportedDeal = (long)ticket;
   }

   FileClose(handle);
}
// Note on entry price: a closing deal knows its own price, not the price
// the position opened at. QuantEngine compares realised profit per trade,
// which the closing deal does carry, so the field is written as 0 rather
// than guessed from an unrelated deal.

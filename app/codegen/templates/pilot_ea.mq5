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
input string OutputFile    = "quantengine_live_v2.csv";   // written to the COMMON Files folder
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
   // `broker` is what makes REQ-AUTO-3 possible: the same strategy losing
   // on every broker is alpha decay, losing on one is a broker problem.
   // `gmt_offset_seconds` turns closed_at (server clock) into UTC.
   FileWrite(handle, "deal_ticket", "closed_at", "magic", "symbol", "volume", "profit", "entry_price", "exit_price",
             "broker", "gmt_offset_seconds", "position_id");
   FileClose(handle);
}

// True when `ticket` is the last closing deal of its position in the
// selected history - a PARTIAL TP closes a position in two deals, and it
// is one trade.
bool IsFinalExit(ulong ticket, long positionId, int total)
{
   for(int j = 0; j < total; j++)
   {
      ulong other = HistoryDealGetTicket(j);
      if(other <= ticket) continue;
      if(HistoryDealGetInteger(other, DEAL_POSITION_ID) != positionId) continue;
      long entry = HistoryDealGetInteger(other, DEAL_ENTRY);
      if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY) return false;
   }
   return true;
}

void ScanHistory()
{
   datetime from = TimeCurrent() - (datetime)LookbackDays * 86400;
   if(!HistorySelect(from, TimeCurrent())) return;

   int handle = FileOpen(OutputFile, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE) return;
   FileSeek(handle, 0, SEEK_END);

   int gmtOffset = (int)(TimeTradeServer() - TimeGMT());
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if((long)ticket <= LastReportedDeal) continue;

      long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      if(magic < QE_MAGIC_MIN || magic > QE_MAGIC_MAX) continue;           // not ours
      long entryKind = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entryKind != DEAL_ENTRY_OUT && entryKind != DEAL_ENTRY_OUT_BY) continue;  // closes only

      // One row per position, written when it is fully closed. Each
      // partial close used to be its own "trade", which doubled the trade
      // count of every PARTIAL TP strategy and inflated its win rate.
      long positionId = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      if(PositionSelectByTicket((ulong)positionId)) continue;   // still open: a partial close
      if(!IsFinalExit(ticket, positionId, total)) continue;

      // Net of commission and swap, on every deal of the position - what
      // the account actually made. DEAL_PROFIT alone left both out, so a
      // live run looked better than the account statement.
      double net = 0.0, entryPrice = 0.0, volume = 0.0;
      for(int j = 0; j < total; j++)
      {
         ulong deal = HistoryDealGetTicket(j);
         if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) != positionId) continue;
         net += HistoryDealGetDouble(deal, DEAL_PROFIT) + HistoryDealGetDouble(deal, DEAL_COMMISSION)
              + HistoryDealGetDouble(deal, DEAL_SWAP);
         if(HistoryDealGetInteger(deal, DEAL_ENTRY) == DEAL_ENTRY_IN)
         {
            entryPrice = HistoryDealGetDouble(deal, DEAL_PRICE);
            volume = HistoryDealGetDouble(deal, DEAL_VOLUME);
         }
      }

      FileWrite(
         handle,
         (string)ticket,
         TimeToString((datetime)HistoryDealGetInteger(ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
         (string)magic,
         HistoryDealGetString(ticket, DEAL_SYMBOL),
         DoubleToString(volume, 2),
         DoubleToString(net, 2),
         DoubleToString(entryPrice, 5),
         DoubleToString(HistoryDealGetDouble(ticket, DEAL_PRICE), 5),
         AccountInfoString(ACCOUNT_SERVER),
         (string)gmtOffset,
         (string)positionId
      );

      if((long)ticket > LastReportedDeal) LastReportedDeal = (long)ticket;
   }

   FileClose(handle);
}

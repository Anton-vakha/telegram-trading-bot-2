
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import threading
import time
import os
import requests
from collections import deque
from statistics import mean
from datetime import datetime, timedelta

# =========================
# Config
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_TOKEN")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "YOUR_TWELVEDATA_KEY")

# Respect TwelveData free tier: <= 8 requests/minute
RATE_LIMIT_PER_MIN = int(os.getenv("TD_RATE_LIMIT_PER_MIN", "8"))
RATE_WINDOW_SEC = 60

# =========================
# Globals
# =========================
enabled = False
symbols = ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","EUR/JPY","GBP/JPY","AUD/JPY"]
chat_id = None

signal_interval_sec = 1

last_checked_ts = {}
last_m1_rsi_state = {}
last_signal = {}
signal_stats = {"total": 0, "buy": 0, "sell": 0, "weak": 0}

# =========================
# Rate Limiter
# =========================
class MinuteRateLimiter:
    def __init__(self, max_per_min=8, window_sec=60):
        self.max = max_per_min
        self.window = window_sec
        self.calls = deque()
        self.lock = threading.Lock()

    def wait_for_slot(self):
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] >= self.window:
                    self.calls.popleft()
                if len(self.calls) < self.max:
                    self.calls.append(now)
                    return
                sleep_for = self.window - (now - self.calls[0])
            time.sleep(max(0.05, sleep_for))

rate_limiter = MinuteRateLimiter(RATE_LIMIT_PER_MIN, RATE_WINDOW_SEC)

def throttled_get(url, params, timeout=15):
    rate_limiter.wait_for_slot()
    r = requests.get(url, params=params, timeout=timeout)
    return r

# =========================
# Utilities
# =========================
def fetch_candles(pair: str, interval="1min", size=200):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": size,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
        "order": "ASC"
    }
    r = throttled_get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Bad response from TwelveData: {data}")
    candles = []
    for v in data["values"]:
        candles.append({
            "datetime": v["datetime"],
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"])
        })
    candles.sort(key=lambda x: x["datetime"])
    return candles

# =========================
# Bot Commands
# =========================
def start(update: Update, context: CallbackContext):
    global enabled, chat_id
    enabled = True
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id=chat_id, text=f"✅ Сигналы включены. Лимит {RATE_LIMIT_PER_MIN}/мин. Пары: {', '.join(symbols)}")

def stop(update: Update, context: CallbackContext):
    global enabled
    enabled = False
    context.bot.send_message(chat_id=update.effective_chat.id, text="🛑 Сигналы отключены.")

def stats(update: Update, context: CallbackContext):
    total = signal_stats["total"]
    buy = signal_stats["buy"]
    sell = signal_stats["sell"]
    weak = signal_stats["weak"]
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(f"📊 Всего: {total}\n🟢 Buy: {buy}\n🔴 Sell: {sell}\n🟡 Weak: {weak}")
    )

# =========================
# Analyzer Loop
# =========================
def analyze(context_bot_send):
    global enabled
    while True:
        try:
            if enabled and chat_id:
                for pair in symbols:
                    try:
                        candles = fetch_candles(pair, "1min", 50)
                        last = candles[-1]
                        msg = f"💱 {pair}\nЦена: {last['close']}"
                        context_bot_send(chat_id=chat_id, text=msg)
                    except Exception as e:
                        context_bot_send(chat_id=chat_id, text=f"⚠️ Ошибка {pair}: {e}")
                time.sleep(signal_interval_sec)
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(2)

# =========================
# Main
# =========================
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("stop", stop))
    dp.add_handler(CommandHandler("stats", stats))

    threading.Thread(target=lambda: analyze(updater.bot.send_message), daemon=True).start()
    updater.start_polling(drop_pending_updates=True)
    updater.idle()

if __name__ == "__main__":
    main()

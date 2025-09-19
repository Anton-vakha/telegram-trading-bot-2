# main_full_bot_signals_multi_rl.py
# Telegram FX signals bot with TwelveData rate limiting (<= 8 req/min by default)

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import threading
import time
import os
import re
import requests
from collections import deque
from statistics import mean
from datetime import datetime, timedelta

# =========================
# Config (env vars on Railway)
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

# Respect TwelveData free tier: <= 8 requests/minute (can be increased on paid plan)
RATE_LIMIT_PER_MIN = int(os.getenv("TD_RATE_LIMIT_PER_MIN", "8"))
RATE_WINDOW_SEC = 60

# =========================
# Globals
# =========================
enabled = False
symbols = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "EUR/JPY", "GBP/JPY", "AUD/JPY"
]
chat_id = None

# loop tick; real pacing is controlled by the rate limiter
signal_interval_sec = 1

# simple stats
signal_stats = {"total": 0, "buy": 0, "sell": 0, "weak": 0}

# Strategy thresholds & toggles
cfg = {
    "rsi_period": 14,
    "rsi_buy": 30,
    "rsi_sell": 70,

    "pinbar_body_pct": 0.30,   # body <= 30% of candle
    "pinbar_wick_ratio": 0.66, # wick >= 66% of candle

    "sr_lookback": 60,         # candles to build S/R
    "sr_window": 5,            # local extrema window
    "sr_touches": 2,           # min touches in cluster
    "sr_tolerance": 0.001,     # 0.1% cluster tolerance
    "near_level_pct": 0.0015,  # signal if within 0.15%
}

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
                # drop timestamps older than window
                while self.calls and now - self.calls[0] >= self.window:
                    self.calls.popleft()
                if len(self.calls) < self.max:
                    self.calls.append(now)
                    return
                # time to next free slot
                sleep_for = self.window - (now - self.calls[0])
            time.sleep(max(0.05, sleep_for))

rate_limiter = MinuteRateLimiter(RATE_LIMIT_PER_MIN, RATE_WINDOW_SEC)

def throttled_get(url, params, timeout=15):
    rate_limiter.wait_for_slot()
    r = requests.get(url, params=params, timeout=timeout)
    return r

# =========================
# Market data helpers
# =========================
def fetch_candles(pair: str, interval="1min", size=200):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "outputsize": size,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
        "order": "ASC",
    }
    r = throttled_get(url, params)
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
            "close": float(v["close"]),
        })
    candles.sort(key=lambda x: x["datetime"])
    return candles

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [None] * len(closes)
    rsis = [None] * len(closes)
    gains = [max(closes[i] - closes[i-1], 0.0) for i in range(1, period+1)]
    losses = [max(closes[i-1] - closes[i], 0.0) for i in range(1, period+1)]
    avg_gain = sum(gains)/period
    avg_loss = sum(losses)/period
    rsis[period] = 100 - (100 / (1 + (avg_gain/avg_loss if avg_loss != 0 else float("inf"))))
    for i in range(period+1, len(closes)):
        ch = closes[i] - closes[i-1]
        g = max(ch, 0.0); l = max(-ch, 0.0)
        avg_gain = (avg_gain*(period-1) + g) / period
        avg_loss = (avg_loss*(period-1) + l) / period
        rsis[i] = 100 - (100 / (1 + (avg_gain/avg_loss if avg_loss != 0 else float("inf"))))
    return rsis

def is_pinbar(c):
    hi, lo, op, cl = c["high"], c["low"], c["open"], c["close"]
    full = max(hi - lo, 1e-9)
    body = abs(cl - op)
    up = hi - max(op, cl)
    dn = min(op, cl) - lo
    body_ok = body <= cfg["pinbar_body_pct"] * full
    up_ok = up >= cfg["pinbar_wick_ratio"] * full
    dn_ok = dn >= cfg["pinbar_wick_ratio"] * full
    if body_ok and dn_ok and not up_ok and cl > op:
        return "bullish"
    if body_ok and up_ok and not dn_ok and cl < op:
        return "bearish"
    return None

def find_levels(candles):
    w = cfg["sr_window"]; tol = cfg["sr_tolerance"]
    highs, lows = [], []
    for i in range(w, len(candles)-w):
        ch = candles[i]
        if all(ch["high"] >= candles[j]["high"] for j in range(i-w, i+w+1)):
            highs.append(ch["high"])
        if all(ch["low"]  <= candles[j]["low"]  for j in range(i-w, i+w+1)):
            lows.append(ch["low"])
    def cluster(arr):
        arr = sorted(arr); out = []; bucket = []
        for v in arr:
            if not bucket or abs(v - mean(bucket))/max(mean(bucket), 1e-9) <= tol:
                bucket.append(v)
            else:
                if len(bucket) >= cfg["sr_touches"]:
                    out.append(mean(bucket))
                bucket = [v]
        if bucket and len(bucket) >= cfg["sr_touches"]:
            out.append(mean(bucket))
        return out
    return {"resistance": cluster(highs), "support": cluster(lows)}

def nearest_level(price, levels):
    if not levels:
        return None, None
    L = min(levels, key=lambda x: abs(price - x))
    dist = abs(price - L) / max(price, 1e-9)
    return L, dist

def compute_entry_time(dt_str):
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt, dt + timedelta(minutes=1)
    except Exception:
        return None, None

def format_simple(pair, kind, strength, dt_str):
    _, entry = compute_entry_time(dt_str)
    badge = "🟢" if (kind == "buy" and strength == "strong") else \
            "🔴" if (kind == "sell" and strength == "strong") else "🟡"
    entry_str = entry.strftime("%H:%M") if entry else "next"
    return (
        f"💱 {pair}\n"
        f"{badge} {kind.upper()} ({strength})\n"
        f"⏰ Вход: {entry_str}\n"
        f"⌛ Эксп: 2–3 мин"
    )

# =========================
# Bot commands
# =========================
def start(update: Update, context: CallbackContext):
    global enabled, chat_id
    enabled = True
    chat_id = update.effective_chat.id
    context.bot.send_message(
        chat_id=chat_id,
        text=f"✅ Сигналы включены. Лимит {RATE_LIMIT_PER_MIN}/мин. Пары: {', '.join(symbols)}"
    )

def stop(update: Update, context: CallbackContext):
    global enabled
    enabled = False
    context.bot.send_message(chat_id=update.effective_chat.id, text="🛑 Сигналы отключены.")

def stats(update: Update, context: CallbackContext):
    total = signal_stats["total"]; buy = signal_stats["buy"]; sell = signal_stats["sell"]; weak = signal_stats["weak"]
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"📊 Всего: {total}\n🟢 Buy: {buy}\n🔴 Sell: {sell}\n🟡 Weak: {weak}"
    )

def pairs_cmd(update: Update, context: CallbackContext):
    global symbols
    if not context.args or context.args[0].lower() == "list":
        listing = "\n".join(f"- {s}" for s in symbols)
        update.message.reply_text("Текущий список пар:\n" + listing)
        return
    sub = context.args[0].lower()
    if sub == "add":
        added = []
        for raw in context.args[1:]:
            s = raw.replace("_", "/").upper()
            if s not in symbols:
                symbols.append(s); added.append(s)
        update.message.reply_text("✅ Добавлены: " + (", ".join(added) if added else "(ничего)"))
    elif sub == "remove":
        removed = []
        for raw in context.args[1:]:
            s = raw.replace("_", "/").upper()
            if s in symbols:
                symbols.remove(s); removed.append(s)
        update.message.reply_text("🗑 Удалены: " + (", ".join(removed) if removed else "(ничего)"))
    elif sub == "set":
        new_list = []
        for raw in context.args[1:]:
            s = raw.replace("_", "/").upper()
            if s not in new_list:
                new_list.append(s)
        symbols = new_list or symbols
        update.message.reply_text("✅ Новый список пар: " + ", ".join(symbols))
    elif sub == "clear":
        symbols = []
        update.message.reply_text("Список пар очищен.")
    else:
        update.message.reply_text("Использование: /pairs [list|add|remove|set|clear] EUR_USD GBP/USD ...")

# =========================
# Analyzer loop (signals)
# =========================
def analyze(context_bot_send):
    global enabled
    while True:
        try:
            if enabled and chat_id and symbols:
                for pair in list(symbols):
                    try:
                        candles = fetch_candles(pair, "1min", max(120, cfg["sr_lookback"]+20))
                        if not candles:
                            continue
                        last = candles[-1]
                        closes = [c["close"] for c in candles]
                        rsis = calc_rsi(closes, cfg["rsi_period"])
                        rsi_cur = rsis[-1]
                        if rsi_cur is None:
                            continue

                        pin = is_pinbar(last)
                        lookback = candles[-cfg["sr_lookback"]:] if len(candles) >= cfg["sr_lookback"] else candles
                        levels = find_levels(lookback)
                        price = last["close"]
                        sup, sup_d = nearest_level(price, levels["support"])
                        res, res_d = nearest_level(price, levels["resistance"])
                        near_sup = sup is not None and sup_d is not None and sup_d <= cfg["near_level_pct"]
                        near_res = res is not None and res_d is not None and res_d <= cfg["near_level_pct"]

                        strong = None
                        if pin == "bullish" and near_sup and rsi_cur <= cfg["rsi_buy"]:
                            strong = ("buy", sup, "support")
                        if pin == "bearish" and near_res and rsi_cur >= cfg["rsi_sell"]:
                            strong = ("sell", res, "resistance")

                        weak = None
                        if not strong:
                            bull_tests = [pin == "bullish", near_sup, rsi_cur <= cfg["rsi_buy"] + 5]
                            bear_tests = [pin == "bearish", near_res, rsi_cur >= cfg["rsi_sell"] - 5]
                            if sum(bull_tests) >= 2:
                                weak = ("buy", sup if sup else price, "support" if sup else "area")
                            if sum(bear_tests) >= 2:
                                weak = ("sell", res if res else price, "resistance" if res else "area")

                        chosen = None
                        if strong:
                            chosen = ("strong",) + strong
                        elif weak:
                            chosen = ("weak",) + weak

                        if chosen:
                            strength, kind, _level, _lt = chosen
                            msg = format_simple(pair, kind, strength, last["datetime"])
                            context_bot_send(chat_id=chat_id, text=msg)

                            signal_stats["total"] += 1
                            if strength == "weak":
                                signal_stats["weak"] += 1
                            if kind == "buy":
                                signal_stats["buy"] += 1
                            else:
                                signal_stats["sell"] += 1

                    except Exception as inner_e:
                        context_bot_send(chat_id=chat_id, text=f"⚠️ Ошибка {pair}: {inner_e}")

                time.sleep(signal_interval_sec)  # limiter governs HTTP pace
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2)

# =========================
# Main
# =========================
def main():
    # quick sanity check to make debugging easier in logs
    if not TELEGRAM_TOKEN or not re.match(r"^\d+:[A-Za-z0-9_-]{35}$", TELEGRAM_TOKEN):
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing or invalid format")

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("stop", stop))
    dp.add_handler(CommandHandler("stats", stats))
    dp.add_handler(CommandHandler("pairs", pairs_cmd))

    threading.Thread(target=lambda: analyze(updater.bot.send_message), daemon=True).start()

    # drop_pending_updates avoids conflicts after restarts
    updater.start_polling(drop_pending_updates=True)
    updater.idle()

if __name__ == "__main__":
    main()

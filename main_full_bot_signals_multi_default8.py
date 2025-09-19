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

signal_mode = "all"  # "all" | "strong_only" | "strong_cross"

cfg = {
    "rsi_period": 14,
    "rsi_buy": 30,
    "rsi_sell": 70,
    "pinbar_body_pct": 0.30,
    "pinbar_wick_ratio": 0.66,
    "sr_lookback": 60,
    "sr_window": 5,
    "sr_touches": 2,
    "sr_tolerance": 0.001,
    "near_level_pct": 0.0015,
    "use_mtf": 1,
    "mtf_rsi_slope_eps": 0.5,
    "use_m15": 1,
    "m15_rsi_slope_eps": 0.5,
    "rsi_cross_only": 1,
    "weak_use_pinbar": 1,
    "weak_use_level": 1,
    "weak_use_rsi": 1,
    "weak_min_true": 2,
    "mtf_every_n_cycles": 3
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
    url = "https://api.twelvedata.com/ti

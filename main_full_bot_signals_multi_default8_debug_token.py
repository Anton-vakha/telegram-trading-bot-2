
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import threading
import time
import os
import requests
from statistics import mean
from datetime import datetime, timedelta

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_TOKEN")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "be0a977491c54929a004de3a1eed7fbe")

enabled = False
# Default 8 recommended pairs
symbols = ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD","EUR/JPY","GBP/JPY","AUD/JPY"]
chat_id = None
signal_interval_sec = 30

last_checked_ts = {}
last_m1_rsi_state = {}
last_signal = {}

signal_stats = {"total": 0, "buy": 0, "sell": 0, "weak": 0}
signal_mode = "all"

cfg = {
    "rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70,
    "pinbar_body_pct": 0.30, "pinbar_wick_ratio": 0.66,
    "sr_lookback": 60, "sr_window": 5, "sr_touches": 2, "sr_tolerance": 0.001, "near_level_pct": 0.0015,
    "use_mtf": 1, "mtf_rsi_slope_eps": 0.5, "use_m15": 1, "m15_rsi_slope_eps": 0.5,
    "rsi_cross_only": 1,
    "weak_use_pinbar": 1, "weak_use_level": 1, "weak_use_rsi": 1, "weak_min_true": 2
}

def fetch_candles(pair: str, interval="1min", size=200):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": pair, "interval": interval, "outputsize": size, "apikey": TWELVE_DATA_KEY, "format": "JSON", "order": "ASC"}
    r = requests.get(url, params=params, timeout=15); r.raise_for_status()
    data = r.json()
    if "values" not in data: raise RuntimeError(f"Bad response from TwelveData: {data}")
    candles = [{"datetime": v["datetime"], "open": float(v["open"]), "high": float(v["high"]), "low": float(v["low"]), "close": float(v["close"])} for v in data["values"]]
    candles.sort(key=lambda x: x["datetime"]); return candles

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return [None]*len(closes)
    rsis=[None]*len(closes); gains=[]; losses=[]
    for i in range(1, period+1):
        ch=closes[i]-closes[i-1]; gains.append(max(ch,0.0)); losses.append(max(-ch,0.0))
    avg_gain=sum(gains)/period; avg_loss=sum(losses)/period; rs=(avg_gain/avg_loss) if avg_loss!=0 else float('inf')
    rsis[period]=100-(100/(1+rs))
    for i in range(period+1, len(closes)):
        ch=closes[i]-closes[i-1]; gain=max(ch,0.0); loss=max(-ch,0.0)
        avg_gain=(avg_gain*(period-1)+gain)/period; avg_loss=(avg_loss*(period-1)+loss)/period
        rs=(avg_gain/avg_loss) if avg_loss!=0 else float('inf'); rsis[i]=100-(100/(1+rs))
    return rsis

def is_pinbar(c):
    hi,lo,op,cl=c["high"],c["low"],c["open"],c["close"]
    full=max(hi-lo,1e-9); body=abs(cl-op); upper=hi-max(op,cl); lower=min(op,cl)-lo
    body_ok=body<=cfg["pinbar_body_pct"]*full; up_ok=upper>=cfg["pinbar_wick_ratio"]*full; low_ok=lower>=cfg["pinbar_wick_ratio"]*full
    if body_ok and low_ok and not up_ok and cl>op: return "bullish"
    if body_ok and up_ok and not low_ok and cl<op: return "bearish"
    return None

from statistics import mean as _mean
def find_levels(candles):
    highs,lows=[],[]; n=len(candles); w=cfg["sr_window"]; tol=cfg["sr_tolerance"]
    for i in range(w, n-w):
        chw=candles[i]
        if all(chw["high"]>=candles[j]["high"] for j in range(i-w,i+w+1)): highs.append(chw["high"])
        if all(chw["low"] <=candles[j]["low"]  for j in range(i-w,i+w+1)): lows.append(chw["low"])
    def cluster(levels):
        levels=sorted(levels); clusters=[]
        for lvl in levels:
            if not clusters or abs(lvl-_mean(clusters[-1]))/max(_mean(clusters[-1]),1e-9)>tol: clusters.append([lvl])
            else: clusters[-1].append(lvl)
        return [_mean(cl) for cl in clusters if len(cl)>=cfg["sr_touches"]]
    return {"resistance": cluster(highs), "support": cluster(lows)}

def nearest_level(price, levels):
    if not levels: return (None,None)
    nearest=min(levels, key=lambda L: abs(price-L))
    return nearest, abs(price-nearest)/max(price,1e-9)

def rsi_state(v):
    if v is None: return None
    if v<=cfg["rsi_buy"]: return "oversold"
    if v>=cfg["rsi_sell"]: return "overbought"
    return "neutral"

def crossed_out_of_zone(prev_state, curr_val, direction):
    if curr_val is None or prev_state is None: return False
    return (prev_state=="oversold" and curr_val>cfg["rsi_buy"] and direction=="buy") or            (prev_state=="overbought" and curr_val<cfg["rsi_sell"] and direction=="sell")

def rsi_slope_ok(direction, rsis, eps):
    if not rsis or len(rsis)<2 or rsis[-1] is None or rsis[-2] is None: return False
    slope=rsis[-1]-rsis[-2]
    return slope>=eps if direction=="buy" else slope<=-eps

def compute_entry_time(dt_str):
    try:
        dt=datetime.strptime(dt_str,"%Y-%m-%d %H:%M:%S"); return dt, dt+timedelta(minutes=1)
    except Exception: return None,None

def format_simple_alert(pair, kind, strength, dt_str):
    _, entry=compute_entry_time(dt_str)
    badge="🟢" if (strength=="strong" and kind=="buy") else ("🔴" if strength=="strong" and kind=="sell" else "🟡")
    entry_str=entry.strftime("%H:%M") if entry else "next"
    return f"💱 {pair}\n{badge} {kind.upper()} ({strength})\n⏰ Вход: {entry_str}\n⌛ Эксп: 2–3 мин"

def format_detailed(pair, kind, candle, rsi_val, level, level_type, strength="strong", extras=None):
    dt=candle["datetime"]; price=candle["close"]; extra="\n".join(extras) if extras else ""
    base=f"\u23F0 <b>{pair}</b> — {kind.upper()} ({strength})\n\u23F1 {dt}\n\U0001F4C8 RSI: {rsi_val:.1f} (P={cfg['rsi_period']})\n\U0001F6A7 {level_type.title()}: {level:.5f}\n\U0001F4B0 Price: {price:.5f}\n\u23F3 Expiry: 2–3 min"
    return base + (f"\n{extra}" if extra else "")

def start(update: Update, context: CallbackContext):
    global enabled, chat_id; enabled=True; chat_id=update.effective_chat.id
    context.bot.send_message(chat_id=chat_id, text="✅ Сигналы включены. Анализирую 8 пар по кругу.")

def stop(update: Update, context: CallbackContext):
    global enabled; enabled=False
    context.bot.send_message(chat_id=update.effective_chat.id, text="🛑 Сигналы отключены.")

def pair(update: Update, context: CallbackContext):
    global symbols, last_checked_ts, last_m1_rsi_state
    if context.args:
        sym=context.args[0].replace("_","/").upper()
        symbols=[sym]; last_checked_ts={}; last_m1_rsi_state={}; last_signal.clear()
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"🔁 Активная пара: {sym} (список сброшен)")
    else:
        context.bot.send_message(chat_id=update.effective_chat.id, text="❗ Пример: /pair EUR_USD")

def pairs(update: Update, context: CallbackContext):
    global symbols, last_checked_ts, last_m1_rsi_state
    if not context.args:
        listing="\n".join(f"- {s}" for s in symbols) if symbols else "(пусто)"
        update.message.reply_text("Текущий список пар:\n"+listing); return
    sub=context.args[0].lower()
    if sub=="list":
        listing="\n".join(f"- {s}" for s in symbols) if symbols else "(пусто)"
        update.message.reply_text("Текущий список пар:\n"+listing)
    elif sub=="add" and len(context.args)>=2:
        added=[]
        for raw in context.args[1:]:
            s=raw.replace("_","/").upper()
            if s not in symbols: symbols.append(s); added.append(s)
        update.message.reply_text("✅ Добавлены: "+", ".join(added) if added else "Ничего не добавлено.")
    elif sub=="remove" and len(context.args)>=2:
        removed=[]
        for raw in context.args[1:]:
            s=raw.replace("_","/").upper()
            if s in symbols:
                symbols.remove(s); last_checked_ts.pop(s,None); last_m1_rsi_state.pop(s,None); last_signal.pop(s,None); removed.append(s)
        update.message.reply_text("🗑 Удалены: "+", ".join(removed) if removed else "Ничего не удалено.")
    elif sub=="set" and len(context.args)>=2:
        new=[]
        for raw in context.args[1:]:
            s=raw.replace("_","/").upper()
            if s not in new: new.append(s)
        symbols=new; last_checked_ts={}; last_m1_rsi_state={}; last_signal.clear()
        update.message.reply_text("✅ Установлен новый список пар: "+", ".join(symbols))
    elif sub=="clear":
        symbols=[]; last_checked_ts={}; last_m1_rsi_state={}; last_signal.clear()
        update.message.reply_text("Список пар очищен.")
    else:
        update.message.reply_text("Использование: /pairs [list|add|remove|set|clear] ...")

def stat(update: Update, context: CallbackContext):
    listing="\n".join(f"- {s}" for s in symbols) if symbols else "(пусто)"
    context.bot.send_message(chat_id=update.effective_chat.id,
        text=(f"📊 Сигналы: всего {signal_stats['total']}, buy {signal_stats['buy']}, sell {signal_stats['sell']}, weak {signal_stats['weak']}\n\n"
              f"Пары в анализе:\n{listing}"))

def help_command(update: Update, context: CallbackContext):
    txt=(
        "<b>По умолчанию анализирую 8 пар:</b> EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, EUR/JPY, GBP/JPY, AUD/JPY\n\n"
        "Команды:\n"
        "/start • /stop\n"
        "/pairs list|add|remove|set|clear\n"
        "/pair EUR_USD — задать одну пару (сброс списка)\n"
        "/stats — статистика и список пар\n"
        "/thresholds — пороги стратегии\n"
        "/details EUR_USD — подробности по последнему сигналу пары\n\n"
        "Формат оповещения:\n"
        "💱 EUR/USD\n🟢 BUY (strong)\n⏰ Вход: 21:32\n⌛ Эксп: 2–3 мин"
    )
    context.bot.send_message(chat_id=update.effective_chat.id, text=txt, parse_mode='HTML')

def thresholds(update: Update, context: CallbackContext):
    if not context.args:
        txt=(
            "<b>Пороги:</b> RSI P={rsi_period}, buy/sell={rsi_buy}/{rsi_sell}\n"
            "Pin-bar body<= {pb}% wick>= {pw}%\n"
            "S/R lookback={lb}, window={w}, touches={t}, tol={tol:.2f}%\n"
            "Near-level={nl:.2f}% | M5={m5} eps={m5e} | M15={m15} eps={m15e}\n"
            "RSI-cross-only={cross} | Weak: pinbar={wp} level={wl} rsi={wr} min_true={wm}".format(
                rsi_period=cfg['rsi_period'], rsi_buy=cfg['rsi_buy'], rsi_sell=cfg['rsi_sell'],
                pb=int(cfg['pinbar_body_pct']*100), pw=int(cfg['pinbar_wick_ratio']*100),
                lb=cfg['sr_lookback'], w=cfg['sr_window'], t=cfg['sr_touches'], tol=cfg['sr_tolerance']*100,
                nl=cfg['near_level_pct']*100,
                m5=cfg['use_mtf'], m5e=cfg['mtf_rsi_slope_eps'],
                m15=cfg['use_m15'], m15e=cfg['m15_rsi_slope_eps'],
                cross=cfg['rsi_cross_only'],
                wp=cfg['weak_use_pinbar'], wl=cfg['weak_use_level'], wr=cfg['weak_use_rsi'], wm=cfg['weak_min_true']
            )
        )
        update.message.reply_text(txt, parse_mode='HTML'); return
    changed=[]
    for arg in context.args:
        if "=" not in arg: continue
        k,v=arg.split("=",1); k=k.strip(); v=v.strip()
        if k not in cfg: continue
        try:
            if v.lower() in ("true","false"): val=1 if v.lower()=="true" else 0
            elif "." in v or "e" in v.lower(): val=float(v)
            else: val=int(v)
            cfg[k]=val; changed.append(f"{k}={val}")
        except: pass
    update.message.reply_text("✅ Обновлено: "+", ".join(changed) if changed else "Ничего не обновлено.")

def details(update: Update, context: CallbackContext):
    pair=None
    if context.args: pair=context.args[0].replace("_","/").upper()
    if not pair:
        if symbols: pair=symbols[0]
        else: update.message.reply_text("Нет пар. Пример: /details EUR_USD"); return
    ls=last_signal.get(pair)
    if not ls: update.message.reply_text(f"Нет деталей для {pair}."); return
    msg=format_detailed(ls["pair"], ls["kind"], ls["candle"], ls["rsi"], ls["level"], ls["level_type"], strength=ls["strength"], extras=ls.get("extras", []))
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode='HTML')

def analyze_symbol(pair, context_bot_send):
    m1=fetch_candles(pair, "1min", size=max(200, cfg["sr_lookback"]+50))
    if not m1: return
    latest=m1[-1]
    if last_checked_ts.get(pair)==latest["datetime"]: return

    closes=[c["close"] for c in m1]; rsis=calc_rsi(closes, cfg["rsi_period"])
    rsi_cur=rsis[-1]; rsi_prev=rsis[-2] if len(rsis)>=2 else None

    rsis_m5=rsis_m15=None
    if cfg["use_mtf"]:
        m5=fetch_candles(pair,"5min",size=120); rsis_m5=calc_rsi([c["close"] for c in m5], cfg["rsi_period"])
    if cfg["use_m15"]:
        m15=fetch_candles(pair,"15min",size=120); rsis_m15=calc_rsi([c["close"] for c in m15], cfg["rsi_period"])

    if rsi_cur is None or rsi_prev is None:
        last_checked_ts[pair]=latest["datetime"]; return

    pin=is_pinbar(latest)
    lookback=m1[-cfg["sr_lookback"]:] if len(m1)>=cfg["sr_lookback"] else m1
    levels=find_levels(lookback)

    price=latest["close"]
    sup_near,sup_dist=nearest_level(price, levels["support"])
    res_near,res_dist=nearest_level(price, levels["resistance"])
    near_sup = sup_near is not None and sup_dist is not None and sup_dist<=cfg["near_level_pct"]
    near_res = res_near is not None and res_dist is not None and res_dist<=cfg["near_level_pct"]

    prev=last_m1_rsi_state.get(pair) or rsi_state(rsi_prev)
    curr=rsi_state(rsi_cur)

    strong=None
    if pin=="bullish" and near_sup and rsi_cur<=cfg["rsi_buy"]: strong=("buy", sup_near, "support")
    if pin=="bearish" and near_res and rsi_cur>=cfg["rsi_sell"]: strong=("sell", res_near, "resistance")

    weak=None
    if not strong:
        conds_bull=[]; conds_bear=[]
        if cfg["weak_use_pinbar"]: conds_bull.append(pin=="bullish"); conds_bear.append(pin=="bearish")
        if cfg["weak_use_level"]: conds_bull.append(near_sup); conds_bear.append(near_res)
        if cfg["weak_use_rsi"]: conds_bull.append(rsi_cur<=cfg["rsi_buy"]+5); conds_bear.append(rsi_cur>=cfg["rsi_sell"]-5)
        if sum(1 for c in conds_bull if c)>=cfg["weak_min_true"]: weak=("buy", sup_near if sup_near else price, "support" if sup_near else "area")
        if sum(1 for c in conds_bear if c)>=cfg["weak_min_true"]: weak=("sell", res_near if res_near else price, "resistance" if res_near else "area")

    def cross_ok(direction):
        if signal_mode=="strong_cross": return crossed_out_of_zone(prev, rsi_cur, direction)
        if not cfg["rsi_cross_only"]: return True
        return crossed_out_of_zone(prev, rsi_cur, direction)

    def mtf_ok(direction):
        if cfg["use_mtf"] and not rsi_slope_ok(direction, rsis_m5, cfg["mtf_rsi_slope_eps"]): return False
        if cfg["use_m15"] and not rsi_slope_ok(direction, rsis_m15, cfg["m15_rsi_slope_eps"]): return False
        return True

    chosen=None
    if strong and cross_ok(strong[0]) and mtf_ok(strong[0]): chosen=("strong",)+strong
    elif signal_mode=="all" and weak and cross_ok(weak[0]) and mtf_ok(weak[0]): chosen=("weak",)+weak

    if chosen:
        strength,kind,level,level_type=chosen
        extras=[]
        if cfg["use_mtf"]: extras.append("MTF: M5 согласование")
        if cfg["use_m15"]: extras.append("MTF: M15 согласование")
        if signal_mode=="strong_cross" or cfg["rsi_cross_only"]: extras.append("RSI-cross: выход из зоны подтверждён")
        last_signal[pair]={"pair":pair,"kind":kind,"strength":strength,"candle":latest,"rsi":rsi_cur,"level":level,"level_type":level_type,"extras":extras}
        context_bot_send(chat_id=chat_id, text=format_simple_alert(pair, kind, strength, latest["datetime"]), parse_mode='HTML')
        signal_stats["total"]+=1; signal_stats["weak"]+=1 if strength=="weak" else 0
        if kind=="buy": signal_stats["buy"]+=1
        if kind=="sell": signal_stats["sell"]+=1

    last_checked_ts[pair]=latest["datetime"]
    last_m1_rsi_state[pair]=curr

def analyze(context_bot_send):
    global enabled
    while True:
        try:
            if enabled and chat_id is not None and symbols:
                for pair in list(symbols):
                    try: analyze_symbol(pair, context_bot_send)
                    except Exception as ie: context_bot_send(chat_id=chat_id, text=f"⚠️ Ошибка анализа {pair}: {ie}")
                time.sleep(signal_interval_sec)
            else:
                time.sleep(1)
        except Exception as e:
            try:
                if enabled and chat_id is not None: context_bot_send(chat_id=chat_id, text=f"⚠️ Ошибка анализа: {e}")
            except Exception: pass
            time.sleep(signal_interval_sec)


def main():
    # --- DEBUG: check TELEGRAM_BOT_TOKEN env early ---
    import os, re
    raw = os.getenv("TELEGRAM_BOT_TOKEN", "")
    print(f"[DEBUG] TELEGRAM_BOT_TOKEN present: {bool(raw)} | length={len(raw)}")
    if raw:
        print(f"[DEBUG] startswith={raw[:6]!r}, contains_colon={':' in raw}")
    token = raw.strip()
    # basic regex from PTB for fast validation
    if not re.match(r"^\d+:[A-Za-z0-9_-]{35}$", token or ""):
        print("[DEBUG] Token failed local regex validation (missing? wrong name? has spaces/newlines?).")
        raise SystemExit("ENV TELEGRAM_BOT_TOKEN is missing or invalid. Please set it in Railway → Variables.")
    # --------------------------------------------------

    updater=Updater(TELEGRAM_TOKEN, use_context=True); dp=updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("stop", stop))
    dp.add_handler(CommandHandler("pair", pair))
    dp.add_handler(CommandHandler("pairs", pairs))
    dp.add_handler(CommandHandler("stats", stat))
    dp.add_handler(CommandHandler("thresholds", thresholds))
    dp.add_handler(CommandHandler("details", details))
    dp.add_handler(CommandHandler("help", help_command))
    threading.Thread(target=lambda: analyze(updater.bot.send_message), daemon=True).start()
    updater.start_polling(); updater.idle()

if __name__=="__main__": main()

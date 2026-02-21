"""
Phase A ステップ3: 1h足取得プロトタイプ検証
四季報_260102.csv の上位5銘柄（135A を含む）で
5価格（price_open / price_1130 / price_1230 / price_close / price_nextopen）が
正しく取得できることを確認するスクリプト。
"""

import re
import time
import warnings
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CSV_PATH = "date/row_date/四季報_260102.csv"
ANNOUNCE_YEAR = 2026  # 「2月19日」のような月日表記に付与する年


def parse_date(date_str: str, year: int) -> date:
    """'2月19日' → date(2026, 2, 19)"""
    m = re.match(r"(\d+)月(\d+)日", str(date_str).strip())
    if not m:
        raise ValueError(f"日付パース失敗: {date_str!r}")
    return date(year, int(m.group(1)), int(m.group(2)))


def fetch_prices(code: str, announce_date: date) -> dict:
    """
    指定銘柄・発表日の5価格を yfinance から取得して返す。
    price_open    : 発表日 09:00 バーの Open
    price_1130    : 発表日 11:00 バーの Close（前場引け近似）
    price_1230    : 発表日 12:00 バーの Open（後場始値近似）
    price_close   : 発表日終値（日次データで取得、精度優先）
    price_nextopen: 翌営業日 09:00 バーの Open（週末・祝日自動対応）
    """
    ticker = yf.Ticker(f"{code}.T")

    # ---- 発表日 1h足 ----
    df_1h = ticker.history(
        start=announce_date,
        end=announce_date + timedelta(days=1),
        interval="1h",
    )
    if df_1h.empty:
        raise ValueError("発表日 1h足データなし")

    price_open = float(df_1h.iloc[0]["Open"])

    # 11:00 バー優先、なければ 12:00 より前の最後のバーを使用（低出来高銘柄対応）
    h11 = df_1h[df_1h.index.hour == 11]
    if not h11.empty:
        price_1130 = float(h11["Close"].iloc[-1])
    else:
        pre_noon = df_1h[df_1h.index.hour < 12]
        if pre_noon.empty:
            raise ValueError("発表日 前場データなし（price_1130 取得不可）")
        price_1130 = float(pre_noon["Close"].iloc[-1])
        print(f"  ※ 11時バーなし → {pre_noon.index[-1].hour}時バーClose を price_1130 として使用")

    h12 = df_1h[df_1h.index.hour == 12]
    if h12.empty:
        raise ValueError("発表日 12時バーなし（price_1230 取得不可）")
    price_1230 = float(h12["Open"].iloc[0])

    # ---- 発表日終値（日次データ） ----
    df_daily = ticker.history(
        start=announce_date,
        end=announce_date + timedelta(days=1),
    )
    if df_daily.empty:
        raise ValueError("発表日 日次データなし")
    price_close = float(df_daily["Close"].iloc[0])

    # ---- 翌営業日始値（最大7日先まで取得） ----
    df_next = ticker.history(
        start=announce_date + timedelta(days=1),
        end=announce_date + timedelta(days=7),
        interval="1h",
    )
    if df_next.empty:
        raise ValueError("翌営業日 1h足データなし（price_nextopen 取得不可）")
    price_nextopen = float(df_next.iloc[0]["Open"])
    next_trade_date = df_next.index[0].date()

    return {
        "price_open":     round(price_open, 2),
        "price_1130":     round(price_1130, 2),
        "price_1230":     round(price_1230, 2),
        "price_close":    round(price_close, 2),
        "price_nextopen": round(price_nextopen, 2),
        "next_trade_date": str(next_trade_date),
    }


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # 上位5銘柄（135A を含む先頭5行）
    top5 = df.head(5)

    print("=" * 70)
    print("Phase A ステップ3: 1h足取得プロトタイプ検証")
    print(f"対象: 上位5銘柄  /  CSV: {CSV_PATH}")
    print("=" * 70)

    results = []
    for _, row in top5.iterrows():
        code = str(row["証券コード"]).strip()
        name = str(row["社名"]).strip()
        announce_str = str(row["発表"]).strip()

        try:
            announce_date = parse_date(announce_str, ANNOUNCE_YEAR)
            print(f"\n[{code}] {name}  (発表日: {announce_date})")

            prices = fetch_prices(code, announce_date)

            print(f"  price_open    : {prices['price_open']:>10.2f}  (発表日始値)")
            print(f"  price_1130    : {prices['price_1130']:>10.2f}  (前場引け近似)")
            print(f"  price_1230    : {prices['price_1230']:>10.2f}  (後場始値近似)")
            print(f"  price_close   : {prices['price_close']:>10.2f}  (発表日終値)")
            print(
                f"  price_nextopen: {prices['price_nextopen']:>10.2f}"
                f"  (翌営業日始値, {prices['next_trade_date']})"
            )
            results.append({"code": code, "name": name, "status": "OK", **prices})

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"code": code, "name": name, "status": f"ERROR: {e}"})

        time.sleep(0.5)

    # ---- 結果サマリー ----
    print("\n" + "=" * 70)
    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"結果: {ok_count}/{len(results)} 銘柄で5価格取得成功")

    if ok_count == len(results):
        print("-> Phase A ステップ3 完了: 全銘柄で5価格取得OK")
    else:
        print("-> 一部取得失敗:")
        for r in results:
            if r["status"] != "OK":
                print(f"   [{r['code']}] {r['name']}: {r['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

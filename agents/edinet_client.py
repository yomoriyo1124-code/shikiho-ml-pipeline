"""
EDINET APIクライアント (Option B)

EDINET API v2 を使用して四半期/有価証券報告書の XBRL から以下を取得する:
  - 営業CF / 投資CF / 財務CF / FCF
  - 自己資本比率（正確値）
  - 研究開発費（絶対額）
  - 設備投資額（絶対額）

設計方針:
  - 日付ごとの書類一覧を一括スキャン（ 60 日分 = 60 API 呼び出し）してインデックスを構築
  - インデックスとXBRL解析結果を JSON キャッシュ（90日TTL）に保存し二重呼び出しを防ぐ

使用例:
  from agents.edinet_client import EDINETClient
  client = EDINETClient(api_key="...")
  data = client.get_company_financials("1301", announce_date)
"""

import io
import json
import logging
import time
import warnings
import zipfile
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

EDINET_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
SCAN_DAYS = 180        # 発表日から遡って検索する最大日数（半期報告書の提出は最大6ヶ月前）
CACHE_TTL_DAYS = 90    # キャッシュ有効期間

DOC_CACHE_PATH = Path("date/processed/edinet_doc_cache.json")
FIN_CACHE_PATH = Path("date/processed/edinet_financial_cache.json")

# 対象書類種別（EDINET API v2 実際のコード体系）
# 120=有価証券報告書 / 160=半期報告書
# 130=訂正有価証券報告書 / 170=訂正半期報告書 / 150=訂正四半期報告書
TARGET_DOC_TYPES = {"120", "160", "130", "170", "150"}

# XBRL 要素名マッピング（local-name で検索、優先順に列挙）
# 日本の EDINET XBRL タクソノミは独自サフィックス（InvCF / OpeCF / FinCF / SGA 等）を使う
XBRL_ELEMENTS = {
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromOperatingActivities",
        "NetCashProvidedByOperatingActivities",
    ],
    "investing_cf": [
        # 日本 XBRL 実際の要素名（"Investment" 単数形）
        "NetCashProvidedByUsedInInvestmentActivities",
        # 国際系フォールバック
        "NetCashProvidedByUsedInInvestingActivities",
        "CashFlowsFromInvestingActivities",
        "NetCashUsedInInvestingActivities",
    ],
    "financing_cf": [
        "NetCashProvidedByUsedInFinancingActivities",
        "CashFlowsFromFinancingActivities",
        "NetCashUsedInFinancingActivities",
    ],
    "capex": [
        # 日本 XBRL 実際の要素名（InvCF サフィックス）
        "PurchaseOfPropertyPlantAndEquipmentInvCF",
        "PurchaseOfNoncurrentAssetsInvCF",           # 非流動資産一括（極洋等）
        # 国際系フォールバック
        "PurchaseOfPropertyPlantAndEquipmentAndIntangibleAssets",
        "PurchaseOfPropertyPlantAndEquipment",
        "AcquisitionOfPropertyPlantAndEquipment",
        "CapitalExpendituresPaidDirectly",
    ],
    "rd_expense": [
        # 日本 XBRL 実際の要素名（SGA サフィックス）
        "ExperimentAndResearchExpensesSGA",
        "ResearchAndDevelopmentExpensesSGA",
        # 国際系フォールバック
        "ResearchAndDevelopmentExpenses",
        "ResearchAndDevelopmentCosts",
        "ResearchAndDevelopmentExpensesInCostOfSales",
    ],
    "total_assets": [
        "Assets",
        "TotalAssets",
    ],
    "net_assets": [
        "NetAssets",
        "Equity",
        "TotalEquity",
        "TotalStockholdersEquity",
    ],
    "net_sales": [
        "NetSales",
        "Revenue",
        "Revenues",
        "SalesAndOtherOperatingRevenues",
    ],
    "equity_ratio": [
        "EquityToAssetRatio",
        "EquityRatio",
    ],
}


class EDINETClient:
    """EDINET API v2 クライアント（書類検索・XBRL 解析・キャッシュ管理）"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._doc_cache: dict = self._load_json(DOC_CACHE_PATH)
        self._fin_cache: dict = self._load_json(FIN_CACHE_PATH)

    # ============================================================
    # キャッシュ I/O
    # ============================================================

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ============================================================
    # HTTP
    # ============================================================

    def _get(self, endpoint: str, params: dict, as_bytes: bool = False):
        """EDINET API GET。失敗時は最大 3 回リトライ。"""
        params = {**params, "Subscription-Key": self.api_key}
        url = f"{EDINET_BASE_URL}/{endpoint}"
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.content if as_bytes else resp.json()
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1)
        raise RuntimeError(f"EDINET API 失敗 (3回): {endpoint}")

    # ============================================================
    # 書類インデックス構築（一括スキャン）
    # ============================================================

    def _get_docs_for_date(self, date_str: str) -> list:
        """指定日の書類一覧を取得（日付単位でキャッシュ）。"""
        cache_key = f"d:{date_str}"
        if cache_key in self._doc_cache:
            return self._doc_cache[cache_key]

        try:
            result = self._get("documents.json", {"date": date_str, "type": 2})
            docs = result.get("results", [])
        except Exception as e:
            logger.debug(f"EDINET {date_str} 取得失敗: {e}")
            docs = []

        self._doc_cache[cache_key] = docs
        self._save_json(DOC_CACHE_PATH, self._doc_cache)
        time.sleep(0.35)   # ~3 req/sec
        return docs

    def build_index_for_period(self, end_date: date, days: int = SCAN_DAYS) -> dict:
        """
        end_date から days 日分をスキャンして
        {証券コード: docID} のインデックスを返す（最新書類を優先）。
        """
        cache_key = f"idx:{end_date.isoformat()}:{days}"
        if cache_key in self._doc_cache:
            return self._doc_cache[cache_key]

        index: dict[str, str] = {}
        logger.info(f"EDINET 書類インデックス構築: {end_date - timedelta(days=days)} 〜 {end_date}")

        for delta in range(days):
            search_date = end_date - timedelta(days=delta)
            docs = self._get_docs_for_date(search_date.strftime("%Y-%m-%d"))
            for doc in docs:
                raw_code = str(doc.get("secCode", "")).strip()
                # EDINET の secCode は末尾に 0 が付く 5 桁形式（例: "13010"）
                sec_code = raw_code[:-1] if len(raw_code) == 5 and raw_code[-1] == "0" else raw_code
                if sec_code and doc.get("docTypeCode") in TARGET_DOC_TYPES:
                    if sec_code not in index:  # 最新（時系列的に先に見つかる）を採用
                        index[sec_code] = doc["docID"]

        self._doc_cache[cache_key] = index
        self._save_json(DOC_CACHE_PATH, self._doc_cache)
        logger.info(f"EDINET インデックス: {len(index)} 件")
        return index

    # ============================================================
    # XBRL ダウンロード & パース
    # ============================================================

    def _download_xbrl(self, doc_id: str) -> bytes | None:
        """docID の XBRL ZIP をダウンロードする。"""
        try:
            content = self._get(f"documents/{doc_id}", {"type": 1}, as_bytes=True)
            return content
        except Exception as e:
            logger.warning(f"XBRL ダウンロード失敗 ({doc_id}): {e}")
            return None

    @staticmethod
    def _extract_xbrl_values(zip_bytes: bytes) -> dict:
        """
        XBRL ZIP から財務値を抽出する（local-name ベースの名前空間非依存検索）。
        返り値: {field_name: raw_value_in_yen}
        """
        raw: dict[str, int | float] = {}

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                xbrl_names = [n for n in zf.namelist() if n.lower().endswith(".xbrl")]
                if not xbrl_names:
                    return raw
                # 最も大きなファイル = メインインスタンス文書
                main_xbrl = max(xbrl_names, key=lambda n: zf.getinfo(n).file_size)
                with zf.open(main_xbrl) as xf:
                    tree = ET.parse(xf)
                    root = tree.getroot()
        except Exception as e:
            logger.warning(f"XBRL ZIP 解析失敗: {e}")
            return raw

        # 要素を local-name でマッピング
        name_to_field: dict[str, str] = {}
        for field, patterns in XBRL_ELEMENTS.items():
            for p in patterns:
                name_to_field[p] = field

        for elem in root.iter():
            local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if local not in name_to_field:
                continue
            field = name_to_field[local]
            if field in raw or not elem.text:
                continue
            text = elem.text.strip()
            try:
                raw[field] = int(text)
            except ValueError:
                try:
                    raw[field] = float(text)
                except ValueError:
                    pass

        return raw

    # ============================================================
    # 特徴量計算
    # ============================================================

    @staticmethod
    def _to_million(val) -> float | None:
        if val is None:
            return None
        return round(float(val) / 1_000_000, 2)

    def _derive_features(self, raw: dict) -> dict:
        """XBRL 生値から ML 特徴量を計算する（単位: 百万円 or %）。"""
        m = self._to_million

        operating_cf = raw.get("operating_cf")
        investing_cf = raw.get("investing_cf")
        capex_raw = raw.get("capex")  # 通常は負値（支出）
        rd = raw.get("rd_expense")
        total_assets = raw.get("total_assets")
        net_assets = raw.get("net_assets")
        net_sales = raw.get("net_sales")
        equity_ratio_raw = raw.get("equity_ratio")

        # 自己資本比率
        if equity_ratio_raw is not None:
            er = float(equity_ratio_raw)
            equity_ratio_exact = round(er if er <= 1 else er / 100.0, 4)
            equity_ratio_exact = round(equity_ratio_exact * 100, 4) if equity_ratio_exact <= 1 else equity_ratio_exact
        elif total_assets and net_assets and float(total_assets) != 0:
            equity_ratio_exact = round(float(net_assets) / float(total_assets) * 100, 4)
        else:
            equity_ratio_exact = None

        # FCF = 営業CF + 設備投資（capex は負値なので足す）
        capex_abs = abs(float(capex_raw)) if capex_raw is not None else None
        if operating_cf is not None and capex_raw is not None:
            free_cf = float(operating_cf) + float(capex_raw)  # capex は負値
        else:
            free_cf = None

        # 売上高に対するレシオ（売上高が取れた場合のみ）
        def ratio(numerator, denominator, digits=4) -> float | None:
            if numerator is not None and denominator and float(denominator) != 0:
                return round(abs(float(numerator)) / abs(float(denominator)) * 100, digits)
            return None

        return {
            "operating_cf_m":    m(operating_cf),
            "investing_cf_m":    m(investing_cf),
            "financing_cf_m":    m(raw.get("financing_cf")),
            "free_cf_m":         m(free_cf),
            "capex_m":           m(capex_abs),
            "rd_expense_m":      m(rd),
            "equity_ratio_exact": equity_ratio_exact,
            "capex_to_sales":    ratio(capex_abs, net_sales),   # % (売上高 EDINET 内で取れれば)
            "rd_to_sales":       ratio(rd, net_sales),           # % (同上)
        }

    # ============================================================
    # 公開インターフェース
    # ============================================================

    def get_company_financials(
        self,
        securities_code: str,
        announce_date: date,
        doc_index: dict | None = None,
    ) -> dict:
        """
        指定銘柄の EDINET 財務指標を返す。

        doc_index: build_index_for_period() の戻り値を渡すとインデックス再構築を省略できる。
        キャッシュ（90日）が有効な場合はキャッシュを返す。
        """
        code = str(securities_code).strip()

        # フィナンシャルキャッシュ確認
        if code in self._fin_cache:
            cached_at = self._fin_cache[code].get("cached_at", "2000-01-01")
            if (date.today() - date.fromisoformat(cached_at)).days < CACHE_TTL_DAYS:
                logger.debug(f"[{code}] EDINET キャッシュ利用")
                return self._fin_cache[code].get("data", {})

        # docID の解決
        if doc_index is None:
            doc_index = self.build_index_for_period(announce_date)

        doc_id = doc_index.get(code)
        # 英字コード（例: 135A）は末尾に 0 を付けた形でも探す
        if not doc_id and not code[-1].isdigit():
            doc_id = doc_index.get(code + "0")
        if not doc_id:
            logger.info(f"[{code}] EDINET 書類未発見（対象期間に提出なし）")
            return {}

        # XBRL ダウンロード & 解析
        zip_bytes = self._download_xbrl(doc_id)
        if not zip_bytes:
            return {}

        raw = self._extract_xbrl_values(zip_bytes)
        if not raw:
            logger.warning(f"[{code}] XBRL 値が抽出できなかった (docID={doc_id})")
            return {}

        features = self._derive_features(raw)
        features["edinet_doc_id"] = doc_id
        logger.info(
            f"[{code}] EDINET 取得成功: "
            f"op_cf={features.get('operating_cf_m')}M, "
            f"fcf={features.get('free_cf_m')}M, "
            f"eq_ratio={features.get('equity_ratio_exact')}%"
        )

        # キャッシュ保存
        self._fin_cache[code] = {"cached_at": date.today().isoformat(), "data": features}
        self._save_json(FIN_CACHE_PATH, self._fin_cache)

        return features

"""
Agent2: ML予測エージェント

入力:
  date/processed/master.csv
  feedback/feedback_前週YYMMDD.json  （初回は不要）

出力:
  models/model_{target}_{YYMMDD}.pkl         （4ターゲット分）
  date/processed/四季報_YYMMDD_predicted.csv
  models/model_summary_YYMMDD.json

使用例:
  python agents/agent2_ml_predictor.py
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── パス定義 ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("date/processed")
MODELS_DIR = Path("models")
FEEDBACK_DIR = Path("feedback")
MASTER_CSV = PROCESSED_DIR / "master.csv"
ENCODING = "utf-8-sig"

# ── ターゲット変数 ─────────────────────────────────────────────────────────────
TARGETS = [
    "target_1130_close",
    "target_1230_close",
    "target_1130_nextopen",
    "target_1230_nextopen",
]

# ── 特徴量から除外するカラム ───────────────────────────────────────────────────
EXCLUDE_COLS: set[str] = {
    "証券コード", "社名", "更新日", "備考", "発表",
    "next_trade_date", "price_1130_note", "edinet_doc_id",
    "price_open", "price_1130", "price_1230", "price_close", "price_nextopen",
    *TARGETS,
}

# ── 閾値 ────────────────────────────────────────────────────────────────────
DIRECTION_THRESHOLD = 0.55
MIN_WEEKS_FOR_CLASSIFICATION = 4
OPTUNA_TRIALS_SMALL = 50
OPTUNA_TRIALS_LARGE = 100
DATA_WEEKS_LARGE = 8
N_CV_SPLITS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# データ読み込み / 前処理
# ══════════════════════════════════════════════════════════════════════════════

def load_feedback() -> dict:
    """最新の feedback JSON を返す。なければ空 dict。"""
    files = sorted(FEEDBACK_DIR.glob("feedback_*.json"))
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as fh:
        fb = json.load(fh)
    logger.info("フィードバック読み込み: %s", files[-1].name)
    return fb


def load_and_prepare() -> tuple[pd.DataFrame, list[str]]:
    """master.csv を読み込み、エンコード・型変換を行い (df, feat_cols) を返す。"""
    df = pd.read_csv(MASTER_CSV, encoding=ENCODING)
    df["next_trade_date"] = pd.to_datetime(df["next_trade_date"])
    df = df.sort_values("next_trade_date").reset_index(drop=True)

    # カテゴリ変数を LabelEncoding
    for col in ["市場", "sector", "industry"]:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # object 型の数値列（経常修正率 等）を float へ
    exclude = EXCLUDE_COLS | {"next_trade_date"}
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feat_cols = [c for c in df.columns if c not in exclude]
    return df, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# CV 分割（週単位）
# ══════════════════════════════════════════════════════════════════════════════

def week_cv_splits(df_train: pd.DataFrame, n_splits: int = N_CV_SPLITS) -> list:
    """訓練 DataFrame に対して週単位の CV スプリットを返す。
    各要素: (train_positions, val_positions) — 0-based numpy 配列。
    np.isin は Timestamp / datetime64 混在で誤動作するため pd.Series.isin を使用。
    """
    weeks = sorted(df_train["next_trade_date"].unique())
    n = len(weeks)
    if n < 2:
        return []
    min_train = max(1, n - n_splits)
    splits = []
    dtd = df_train["next_trade_date"]  # pd.Series のまま使う
    for i in range(min_train, n):
        tr_pos = np.where(dtd.isin(weeks[:i]).values)[0]
        val_pos = np.where((dtd == weeks[i]).values)[0]
        if len(val_pos) > 0:
            splits.append((tr_pos, val_pos))
    return splits[-n_splits:]


# ══════════════════════════════════════════════════════════════════════════════
# 評価指標
# ══════════════════════════════════════════════════════════════════════════════

def dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def to_5class(y: np.ndarray) -> np.ndarray:
    """回帰値を 5 クラスラベル (0–4) に変換。"""
    out = np.full(len(y), 2, dtype=int)   # default: 横ばい(2)
    out[y > 2.0] = 4
    out[(y >= 0.5) & (y <= 2.0)] = 3
    out[(y >= -2.0) & (y <= -0.5)] = 1
    out[y < -2.0] = 0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 単一モデル train & predict
# ══════════════════════════════════════════════════════════════════════════════

def _impute(X_tr: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X_tr), imp.transform(X_val), imp


def train_predict_reg(
    name: str, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, **p
) -> np.ndarray:
    """指定モデルで回帰学習し予測値を返す。"""
    if name == "ridge":
        X_tr, X_val, _ = _impute(X_tr, X_val)
        return Ridge(alpha=p.get("alpha", 1.0)).fit(X_tr, y_tr).predict(X_val)
    if name == "rf":
        X_tr, X_val, _ = _impute(X_tr, X_val)
        m = RandomForestRegressor(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            random_state=42, n_jobs=-1,
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    if name == "lgbm":
        m = lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 200),
            num_leaves=p.get("num_leaves", 31),
            learning_rate=p.get("learning_rate", 0.1),
            min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 1.0),
            colsample_bytree=p.get("colsample_bytree", 1.0),
            random_state=42, verbose=-1,
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    if name == "xgb":
        m = xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.1),
            subsample=p.get("subsample", 1.0),
            colsample_bytree=p.get("colsample_bytree", 1.0),
            random_state=42, verbosity=0,
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    if name == "catboost":
        m = CatBoostRegressor(
            n_estimators=p.get("n_estimators", 200),
            depth=p.get("depth", 6),
            learning_rate=p.get("learning_rate", 0.1),
            random_state=42, verbose=0,
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    raise ValueError(f"未知のモデル: {name}")


def train_predict_cls(
    name: str, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, **p
) -> np.ndarray:
    """指定モデルで分類学習し予測クラスを返す。"""
    if name == "ridge":
        X_tr, X_val, _ = _impute(X_tr, X_val)
        return RidgeClassifier(alpha=p.get("alpha", 1.0), class_weight="balanced").fit(X_tr, y_tr).predict(X_val)
    if name == "rf":
        X_tr, X_val, _ = _impute(X_tr, X_val)
        m = RandomForestClassifier(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    if name == "lgbm":
        m = lgb.LGBMClassifier(
            n_estimators=p.get("n_estimators", 200),
            num_leaves=p.get("num_leaves", 31),
            learning_rate=p.get("learning_rate", 0.1),
            min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 1.0),
            colsample_bytree=p.get("colsample_bytree", 1.0),
            random_state=42, verbose=-1, class_weight="balanced",
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    if name == "xgb":
        n_cls = len(np.unique(y_tr))
        m = xgb.XGBClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.1),
            subsample=p.get("subsample", 1.0),
            colsample_bytree=p.get("colsample_bytree", 1.0),
            random_state=42, verbosity=0,
            num_class=n_cls, objective="multi:softmax",
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    if name == "catboost":
        m = CatBoostClassifier(
            n_estimators=p.get("n_estimators", 200),
            depth=p.get("depth", 6),
            learning_rate=p.get("learning_rate", 0.1),
            random_state=42, verbose=0, auto_class_weights="Balanced",
        )
        return m.fit(X_tr, y_tr).predict(X_val)
    raise ValueError(f"未知のモデル: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# CV 比較
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAMES = ["ridge", "rf", "lgbm", "xgb", "catboost"]


def cv_compare_reg(
    X: np.ndarray, y: np.ndarray, splits: list
) -> dict[str, dict]:
    """5モデルを CV 評価し {model_name: {mae, direction_accuracy}} を返す。"""
    results: dict[str, dict] = {}
    for name in MODEL_NAMES:
        maes, accs = [], []
        for tr_pos, val_pos in splits:
            try:
                pred = train_predict_reg(name, X[tr_pos], y[tr_pos], X[val_pos])
                maes.append(mean_absolute_error(y[val_pos], pred))
                accs.append(dir_acc(y[val_pos], pred))
            except Exception as exc:
                logger.debug("CV skip %s: %s", name, exc)
        results[name] = {
            "mae": float(np.mean(maes)) if maes else float("inf"),
            "direction_accuracy": float(np.mean(accs)) if accs else 0.0,
        }
        logger.info("  [reg] %s  MAE=%.4f  dir_acc=%.3f", name, results[name]["mae"], results[name]["direction_accuracy"])
    return results


def cv_compare_cls(
    X: np.ndarray, y_cls: np.ndarray, splits: list
) -> dict[str, dict]:
    """5モデルを CV 評価し {model_name: {f1_macro}} を返す。"""
    results: dict[str, dict] = {}
    for name in MODEL_NAMES:
        f1s = []
        for tr_pos, val_pos in splits:
            try:
                pred = train_predict_cls(name, X[tr_pos], y_cls[tr_pos], X[val_pos])
                f1s.append(f1_score(y_cls[val_pos], pred, average="macro", zero_division=0))
            except Exception as exc:
                logger.debug("CV skip %s: %s", name, exc)
        results[name] = {"f1_macro": float(np.mean(f1s)) if f1s else 0.0}
        logger.info("  [cls] %s  F1_macro=%.4f", name, results[name]["f1_macro"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Optuna 最適化
# ══════════════════════════════════════════════════════════════════════════════

def _suggest_params(trial: optuna.Trial, name: str) -> dict:
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if name == "catboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    return {}


def optuna_optimize(
    name: str,
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list,
    n_trials: int,
    hp_hints: dict,
) -> dict:
    """Optuna でハイパーパラメータを最適化し最良パラメータを返す。"""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, name)
        scores = []
        for tr_pos, val_pos in splits:
            try:
                if mode == "regression":
                    pred = train_predict_reg(name, X[tr_pos], y[tr_pos], X[val_pos], **params)
                    scores.append(mean_absolute_error(y[val_pos], pred))
                else:
                    pred = train_predict_cls(name, X[tr_pos], y[tr_pos], X[val_pos], **params)
                    scores.append(-f1_score(y[val_pos], pred, average="macro", zero_division=0))
            except Exception:
                scores.append(float("inf") if mode == "regression" else 1.0)
        return float(np.mean(scores)) if scores else float("inf")

    direction = "minimize"
    study = optuna.create_study(direction=direction)

    # hp_hints をデフォルト試行として追加
    if hp_hints:
        try:
            study.enqueue_trial(hp_hints)
        except Exception:
            pass

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# 最終モデル学習と予測
# ══════════════════════════════════════════════════════════════════════════════

def train_final(
    name: str, mode: str, X_train: np.ndarray, y_train: np.ndarray, best_params: dict
) -> dict:
    """全訓練データで最終モデルを学習し artifacts dict を返す。"""
    artifacts: dict = {"model_type": name, "mode": mode, "params": best_params}

    if name in ("ridge", "rf"):
        imp = SimpleImputer(strategy="median")
        X_fit = imp.fit_transform(X_train)
        artifacts["imputer"] = imp
    else:
        X_fit = X_train

    reg_params = {k: v for k, v in best_params.items() if k in ("n_estimators", "max_depth", "min_samples_leaf")}

    if mode == "regression":
        if name == "ridge":
            m = Ridge(alpha=best_params.get("alpha", 1.0)).fit(X_fit, y_train)
        elif name == "rf":
            m = RandomForestRegressor(**reg_params, random_state=42, n_jobs=-1).fit(X_fit, y_train)
        elif name == "lgbm":
            m = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1).fit(X_fit, y_train)
        elif name == "xgb":
            m = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0).fit(X_fit, y_train)
        elif name == "catboost":
            m = CatBoostRegressor(**best_params, random_state=42, verbose=0).fit(X_fit, y_train)
        else:
            raise ValueError(f"未知のモデル: {name}")
    else:
        if name == "ridge":
            m = RidgeClassifier(alpha=best_params.get("alpha", 1.0), class_weight="balanced").fit(X_fit, y_train)
        elif name == "rf":
            m = RandomForestClassifier(**reg_params, random_state=42, n_jobs=-1, class_weight="balanced").fit(X_fit, y_train)
        elif name == "lgbm":
            m = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1, class_weight="balanced").fit(X_fit, y_train)
        elif name == "xgb":
            n_cls = len(np.unique(y_train))
            m = xgb.XGBClassifier(
                **best_params, random_state=42, verbosity=0,
                num_class=n_cls, objective="multi:softmax",
            ).fit(X_fit, y_train)
        elif name == "catboost":
            m = CatBoostClassifier(**best_params, random_state=42, verbose=0, auto_class_weights="Balanced").fit(X_fit, y_train)
        else:
            raise ValueError(f"未知のモデル: {name}")

    artifacts["model"] = m
    return artifacts


def predict_final(artifacts: dict, X_pred: np.ndarray) -> np.ndarray:
    """学習済み artifacts を使って予測する。"""
    if "imputer" in artifacts:
        X_pred = artifacts["imputer"].transform(X_pred)
    return artifacts["model"].predict(X_pred)


def feature_importance(artifacts: dict, feat_cols: list[str]) -> dict[str, float]:
    """上位 15 特徴量とその重要度を返す。"""
    m = artifacts["model"]
    name = artifacts["model_type"]
    if name == "ridge":
        coef = m.coef_
        # 多クラス分類では coef_ は (n_classes, n_features) の 2D 配列
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef).flatten()
    elif hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    else:
        return {}
    top_k = 15
    idx = np.argsort(imp)[::-1][:top_k]
    total = imp.sum() or 1.0
    return {feat_cols[i]: float(imp[i] / total * 100) for i in idx if i < len(feat_cols)}


# ══════════════════════════════════════════════════════════════════════════════
# ターゲット別パイプライン
# ══════════════════════════════════════════════════════════════════════════════

def process_target(
    target: str,
    df: pd.DataFrame,
    feat_cols: list[str],
    train_mask: pd.Series,
    pred_mask: pd.Series,
    n_total_weeks: int,
    week_str: str,
    feedback: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """1 ターゲットのモデル比較 → Optuna → 最終学習 → 予測を行う。

    Returns:
        summary  : モデルサマリー dict
        preds    : 予測値配列（回帰: float、分類: class int）
        classes  : 5 クラスラベル配列
    """
    logger.info("━" * 60)
    logger.info("ターゲット: %s", target)

    # フィードバック取得
    target_fb = feedback.get("agent2_feedback", {}).get(target, {})
    preferred_raw = target_fb.get("recommended_model", "")
    preferred = preferred_raw.split("_")[0] if preferred_raw else ""
    force_cls = bool(target_fb.get("switch_to_classification", False))
    hp_hints = target_fb.get("hyperparameter_hints", {})

    # データ準備
    df_train = df[train_mask].reset_index(drop=True)
    X_all = df[feat_cols].values.astype(float)
    y_all = df[target].values.astype(float)
    X_train = X_all[train_mask.values]
    y_train = y_all[train_mask.values]
    X_pred = X_all[pred_mask.values]

    # CV 分割
    splits = week_cv_splits(df_train, n_splits=N_CV_SPLITS)
    if not splits:
        logger.warning("  CV 分割できず（訓練週数が不足）。スキップ。")
        dummy = np.zeros(pred_mask.sum())
        return {"model": "none", "mode": "regression", "mae": None, "direction_accuracy": None}, dummy, to_5class(dummy)

    # ── 回帰モデル比較 ─────────────────────────────────────────────
    logger.info("  回帰モデル比較 (%d splits)...", len(splits))
    reg_results = cv_compare_reg(X_train, y_train, splits)

    best_reg_name = min(reg_results, key=lambda m: reg_results[m]["mae"])
    best_dir_acc = reg_results[best_reg_name]["direction_accuracy"]
    best_mae = reg_results[best_reg_name]["mae"]
    if preferred and preferred in reg_results:
        best_reg_name = preferred
        best_dir_acc = reg_results[preferred]["direction_accuracy"]
        best_mae = reg_results[preferred]["mae"]

    logger.info("  回帰ベストモデル: %s (MAE=%.4f, dir_acc=%.3f)", best_reg_name, best_mae, best_dir_acc)

    # ── 回帰 vs 分類の判定 ─────────────────────────────────────────
    use_cls = force_cls or (
        best_dir_acc < DIRECTION_THRESHOLD and n_total_weeks >= MIN_WEEKS_FOR_CLASSIFICATION
    )
    mode = "classification" if use_cls else "regression"

    if mode == "classification":
        logger.info("  分類モードへ切替（dir_acc=%.3f < %.2f）", best_dir_acc, DIRECTION_THRESHOLD)
        y_cls_train = to_5class(y_train)
        cls_results = cv_compare_cls(X_train, y_cls_train, splits)
        best_name = max(cls_results, key=lambda m: cls_results[m]["f1_macro"])
        if preferred and preferred in cls_results:
            best_name = preferred
        best_f1 = cls_results[best_name]["f1_macro"]
        logger.info("  分類ベストモデル: %s (F1_macro=%.4f)", best_name, best_f1)
    else:
        best_name = best_reg_name
        y_cls_train = None

    # ── Optuna 最適化 ──────────────────────────────────────────────
    n_trials = OPTUNA_TRIALS_LARGE if n_total_weeks >= DATA_WEEKS_LARGE else OPTUNA_TRIALS_SMALL
    logger.info("  Optuna最適化: %s × %d試行 ...", best_name, n_trials)
    y_optuna = y_cls_train if mode == "classification" else y_train
    best_params = optuna_optimize(best_name, mode, X_train, y_optuna, splits, n_trials, hp_hints)
    logger.info("  最良パラメータ: %s", best_params)

    # ── 最終モデル学習 ─────────────────────────────────────────────
    logger.info("  最終モデル学習...")
    artifacts = train_final(best_name, mode, X_train, y_optuna, best_params)

    # ── 予測 ──────────────────────────────────────────────────────
    preds = predict_final(artifacts, X_pred)
    if mode == "regression":
        classes = to_5class(preds)
    else:
        classes = preds.astype(int)
        preds = classes.astype(float)

    # ── 特徴量重要度 ───────────────────────────────────────────────
    top_feats = feature_importance(artifacts, feat_cols)

    # ── モデル保存 ─────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{target}_{week_str}.pkl"
    joblib.dump({"artifacts": artifacts, "feat_cols": feat_cols}, model_path)
    logger.info("  モデル保存: %s", model_path)

    # ── サマリー ───────────────────────────────────────────────────
    cv_metrics = reg_results[best_name] if mode == "regression" else {
        "f1_macro": cls_results[best_name]["f1_macro"],
        "direction_accuracy": reg_results[best_name]["direction_accuracy"],
    }
    summary = {
        "model": best_name,
        "mode": mode,
        "mae": float(best_mae) if mode == "regression" else None,
        "direction_accuracy": float(best_dir_acc),
        "f1_macro": float(best_f1) if mode == "classification" else None,
        "best_params": best_params,
        "top_features": top_feats,
        "all_models_reg": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in reg_results.items()},
    }
    return summary, preds, classes


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feedback = load_feedback()
    df, feat_cols = load_and_prepare()

    # 週情報
    weeks = sorted(df["next_trade_date"].unique())
    latest_week = weeks[-1]
    train_weeks = weeks[:-1]
    n_train_weeks = len(train_weeks)
    n_total_weeks = len(weeks)

    week_str = latest_week.strftime("%y%m%d")
    logger.info("最新週: %s（week_str=%s）", latest_week.date(), week_str)
    logger.info("訓練週数: %d、予測対象週: %d銘柄", n_train_weeks, (df["next_trade_date"] == latest_week).sum())

    if n_train_weeks < 1:
        logger.error("訓練データが不足しています（週数=%d）。終了。", n_train_weeks)
        return

    train_mask = df["next_trade_date"].isin(train_weeks)
    pred_mask = df["next_trade_date"] == latest_week

    # ── 4ターゲットを逐次処理 ──────────────────────────────────────
    all_summaries: dict[str, dict] = {}
    pred_df = df[pred_mask].copy()

    for target in TARGETS:
        summary, preds, classes = process_target(
            target=target,
            df=df,
            feat_cols=feat_cols,
            train_mask=train_mask,
            pred_mask=pred_mask,
            n_total_weeks=n_train_weeks,   # CV判定は訓練週数で行う
            week_str=week_str,
            feedback=feedback,
        )
        all_summaries[target] = summary
        pred_df[f"pred_{target}"] = preds
        pred_df[f"rank_class_{target}"] = classes

    # ── best_target 判定 ──────────────────────────────────────────
    def _score(s: dict) -> float:
        return s.get("direction_accuracy") or 0.0

    best_target = max(all_summaries, key=lambda t: _score(all_summaries[t]))
    logger.info("best_target: %s", best_target)

    # ── predicted.csv 保存 ────────────────────────────────────────
    week_raw = latest_week.strftime("%y%m%d")
    pred_csv = PROCESSED_DIR / f"四季報_{week_raw}_predicted.csv"
    pred_df.to_csv(pred_csv, index=False, encoding=ENCODING)
    logger.info("predicted.csv 保存: %s", pred_csv)

    # ── model_summary.json 保存 ───────────────────────────────────
    summary_payload = {
        "week": week_str,
        "best_target": best_target,
        "targets": all_summaries,
    }
    summary_path = MODELS_DIR / f"model_summary_{week_str}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)
    logger.info("model_summary 保存: %s", summary_path)

    logger.info("Agent2 完了。")


if __name__ == "__main__":
    main()

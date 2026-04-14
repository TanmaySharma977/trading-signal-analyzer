"""
ML Decision Engine - Random Forest / XGBoost for signal prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Tuple
from src.utils.logger import logger


class MLEngine:
    """
    ML-based decision engine.
    Trains on historical data to predict BUY/SELL/HOLD.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",  # Handle imbalanced classes
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=10,
                random_state=42,
            )

        logger.info(f"MLEngine initialized with {model_type}")

    def prepare_features(self, df: pd.DataFrame, pattern_score: float = 0,
                         sentiment_score: float = 0) -> pd.DataFrame:
        """
        Build feature matrix from OHLCV + indicators.
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features["price_change_1d"] = df["close"].pct_change(1)
        features["price_change_3d"] = df["close"].pct_change(3)
        features["price_change_5d"] = df["close"].pct_change(5)

        # Volatility
        features["volatility_5d"] = df["close"].rolling(5).std() / df["close"].rolling(5).mean()
        features["volatility_10d"] = df["close"].rolling(10).std() / df["close"].rolling(10).mean()

        # Volume features
        features["volume_change"] = df["volume"].pct_change(1)
        features["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # Candle features
        features["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)
        features["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-8)
        features["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-8)

        # Technical indicators (if available)
        indicator_cols = ["rsi", "macd", "macd_histogram", "atr",
                          "stoch_k", "adx", "obv"]
        for col in indicator_cols:
            if col in df.columns:
                features[col] = df[col]

        # SMA distances
        if "sma_10" in df.columns:
            features["dist_sma10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
        if "sma_20" in df.columns:
            features["dist_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        if "sma_50" in df.columns:
            features["dist_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

        # Bollinger Band position
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_width = df["bb_upper"] - df["bb_lower"]
            features["bb_position"] = (df["close"] - df["bb_lower"]) / (bb_width + 1e-8)

        # External scores
        features["pattern_score"] = pattern_score
        features["sentiment_score"] = sentiment_score

        # Drop NaN rows
        features = features.dropna()

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        features = features.dropna()

        self.feature_columns = features.columns.tolist()
        return features

    def create_labels(self, df: pd.DataFrame, forward_days: int = 5,
                      buy_threshold: float = 0.02, sell_threshold: float = -0.02) -> pd.Series:
        """
        Create target labels based on FUTURE price movement.

        If price goes up > 2% in next 5 days → BUY (1)
        If price goes down > 2% → SELL (-1)
        Otherwise → HOLD (0)
        """
        future_return = df["close"].shift(-forward_days) / df["close"] - 1

        labels = pd.Series(0, index=df.index)  # Default: HOLD
        labels[future_return > buy_threshold] = 1     # BUY
        labels[future_return < sell_threshold] = -1   # SELL

        return labels

    def train(self, df: pd.DataFrame, pattern_score: float = 0,
              sentiment_score: float = 0) -> Dict:
        """
        Train model on historical data.

        Returns metrics dict.
        """
        logger.info("Training ML model...")

        features = self.prepare_features(df, pattern_score, sentiment_score)
        labels = self.create_labels(df)

        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        # Remove rows where label is NaN (last few days)
        valid = labels.notna()
        features = features[valid]
        labels = labels[valid]

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        # Remove any remaining problematic rows
        mask = features.apply(lambda x: x.between(-1e10, 1e10)).all(axis=1)
        features = features[mask]
        labels = labels[features.index]
        
        if len(features) < 50:
            logger.warning("Not enough data to train. Need at least 50 rows.")
            return {"error": "Not enough data"}

        # Time-series split (don't leak future data!)
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []

        for train_idx, test_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_train = labels.iloc[train_idx]
            y_test = labels.iloc[test_idx]

            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model.fit(X_train_scaled, y_train)
            preds = self.model.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            accuracies.append(acc)

        # Final train on all data
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled, labels)
        self.is_trained = True

        avg_accuracy = np.mean(accuracies)
        logger.info(f"Model trained. CV Accuracy: {avg_accuracy:.2%}")

        # Feature importance
        importances = dict(zip(
            self.feature_columns,
            self.model.feature_importances_,
        ))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "cv_accuracy": round(avg_accuracy, 4),
            "cv_scores": [round(a, 4) for a in accuracies],
            "num_samples": len(features),
            "top_features": top_features,
        }

    def predict(self, df: pd.DataFrame, pattern_score: float = 0,
                sentiment_score: float = 0) -> Dict:
        """
        Predict signal for the latest data point.
        """
        if not self.is_trained:
            logger.warning("Model not trained. Training now...")
            self.train(df, pattern_score, sentiment_score)

        features = self.prepare_features(df, pattern_score, sentiment_score)

        if features.empty:
            return {"signal": "HOLD", "confidence": 0, "probabilities": {}}

        last_features = features.iloc[[-1]]
        last_scaled = self.scaler.transform(last_features)

        prediction = self.model.predict(last_scaled)[0]
        probabilities = self.model.predict_proba(last_scaled)[0]
        classes = self.model.classes_

        prob_dict = {int(c): round(float(p), 4) for c, p in zip(classes, probabilities)}

        signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        signal = signal_map.get(prediction, "HOLD")
        confidence = max(probabilities) * 100

        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
            "prediction_raw": int(prediction),
        }

    def save(self, path: str = "src/models/saved_models/ml_model.pkl"):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "features": self.feature_columns,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = "src/models/saved_models/ml_model.pkl"):
        """Load trained model."""
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_columns = data["features"]
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"No model found at {path}")

    
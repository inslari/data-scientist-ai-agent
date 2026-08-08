from src.core.nodes import node
from src.state import AgentState
from src.services.feature_builder import build_features

from src.core.logging_config import get_logger
logger = get_logger(__name__)

@node("feature_engineering_node")
def feature_engineering_node(state: AgentState) -> AgentState:
    """
    Node 5: Makine Öğrenmesi için Veri Hazırlığı.
    Tüm feature engineering mantığı `src.services.feature_builder` içerisine taşındı.
    """

    logger.info(f"[DEBUG FE] data_path = '{state.get('data_path')}'")
    logger.info(f"[DEBUG FE] sql_results len = {len(state.get('sql_results') or [])}")

    result = build_features(state)
    
    if result.get("error"):
        return {
            "error": result["error"],
            "current_step": "feature_engineering_node",
        }
        
    from src.utils.type_utils import clean_np

    logger.info(f"[DEBUG FE OUT] domain_leakage_report var mi: {state.get('domain_leakage_report') is not None}")
    return clean_np({
        # Ağır diziler diske yazıldı; state'te yol taşınır (inline anahtarlar yalnızca
        # disk yazımı başarısızsa dolar — geri-uyum/fallback).
        "X_train_path": result.get("X_train_path"),
        "X_test_path": result.get("X_test_path"),
        "y_train_path": result.get("y_train_path"),
        "y_test_path": result.get("y_test_path"),
        "X_train": result.get("X_train"),
        "X_test": result.get("X_test"),
        "y_train": result.get("y_train"),
        "y_test": result.get("y_test"),
        "feature_names": result.get("feature_names"),
        "target_name": result.get("target_name"),
        "feature_report": result.get("feature_report"),
        "target_definition": result.get("target_definition", state.get("target_definition")),
        # Fairness segment taşıyıcısı: ham grup değerlerinin test satırlarına hizalı
        # snapshot'ı (yol) + hangi kolonların neden taşınmadığı (meta).
        # Ham test taşıyıcısı (fairness + hata analizi okur) — yol + meta.
        "raw_test_path": result.get("raw_test_path", ""),
        "raw_test_meta": result.get("raw_test_meta", {}),
        # Katman 1 wiring: artifact id (Faz B/C servisleri state'ten okur) + durum bloğu.
        "inference_artifact_id": result.get("inference_artifact_id"),
        "inference_artifact": result.get("inference_artifact"),
        # Veri İlişkileri özeti — split anında gerçek train partisyonundan hesaplandı.
        "relations_report": result.get("relations_report", {}),
        "current_step": "feature_engineering_node",
    })
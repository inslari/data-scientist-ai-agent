import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents import intent_discovery_agent
from tests.db_test_utils import init_results_db, save_intent_discovery_result

TEST_DB_PATH = Path(__file__).resolve().parent / "data" / "intent_discovery_results.db"


class IntentDiscoveryAgentIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_results_db(TEST_DB_PATH)

    def test_intent_discovery_agent_node_saves_results_to_db(self):
        fake_response = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "[{\"intent_name\": \"Customer Churn Prediction\", \"intent_type\": \"predictive\"}]"
                }
            ]
        }

        with patch.object(intent_discovery_agent.agent, "invoke", return_value=fake_response):
            state = {
                "schema_graph": {"tables": ["customers"]},
                "db_intelligence": {"capabilities": ["customer_segmentation"]},
                "signals": {"core_entities": ["Customer"]},
                "domain": "Retail",
                "intents": []
            }

            result = intent_discovery_agent.intent_discovery_agent_node(state)
            self.assertEqual(result["intents"], fake_response["messages"][-1]["content"])

            save_intent_discovery_result(TEST_DB_PATH, state, result)

        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT agent_output FROM intent_discovery_results ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertIn("Customer Churn Prediction", row[0])

    @unittest.skipIf(not (Path(__file__).resolve().parent.parent / "data" / "meta.db").exists(), "Gerçek veri tabanı (meta.db) bulunamadı.")
    def test_intent_discovery_agent_node_real_data(self):
        """
        Gerçek veritabanı (meta.db) üzerinden tabloları okur ve mock kullanmadan
        gerçek ajanı çalıştırır. Sonuçları test veritabanına kaydeder.
        """
        meta_db_path = Path(__file__).resolve().parent.parent / "data" / "meta.db"
        
        conn = sqlite3.connect(meta_db_path)
        cursor = conn.cursor()
        
        # Tabloları çek
        try:
            cursor.execute("SELECT table_name FROM table_schemas")
            tables = [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            tables = []

        # Yetenekleri çek
        try:
            cursor.execute("SELECT capability FROM database_capabilities")
            capabilities = [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            capabilities = []
            
        conn.close()

        if not tables:
            self.skipTest("meta.db içinde tablo bulunamadı.")

        # Gerçek meta.db verilerine dayalı state oluştur
        state = {
            "schema_graph": {"tables": tables},
            "db_intelligence": {"capabilities": capabilities},
            "signals": {"core_entities": [t.capitalize() for t in tables]},
            "domain": "Unknown", # Domain dinamik olarak LLM ile de bulunabilir
            "intents": []
        }

        # Mock olmadan gerçek LLM çağrısı
        result = intent_discovery_agent.intent_discovery_agent_node(state)
        
        # Sonucu veritabanına kaydet
        save_intent_discovery_result(TEST_DB_PATH, state, result)

        # Hata vermeden çalışması ve intents dönmesi yeterlidir
        self.assertIn("intents", result)
        self.assertIsNotNone(result["intents"])


if __name__ == "__main__":
    unittest.main()

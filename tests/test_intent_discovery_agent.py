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


if __name__ == "__main__":
    unittest.main()

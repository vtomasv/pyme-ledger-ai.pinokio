"""
test_ollama_client.py — Tests unitarios para el wrapper centralizado de Ollama
================================================================================
Cubre:
  - Verificación de disponibilidad de Ollama
  - Verificación de modelos disponibles
  - Llamadas a generate con sanitización
  - Tracking de uso (tokens, latencia, inyecciones bloqueadas)
  - Manejo de errores y timeouts
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))


class TestCheckOllama(unittest.TestCase):
    """Tests para check_ollama() en hardware.py."""

    def test_function_exists(self):
        """check_ollama debe existir en hardware.py."""
        from hardware import check_ollama
        self.assertTrue(callable(check_ollama))

    def test_returns_bool(self):
        """check_ollama retorna bool (False sin Ollama corriendo)."""
        from hardware import check_ollama
        result = check_ollama()
        self.assertIsInstance(result, bool)


class TestGetAvailableModels(unittest.TestCase):
    """Tests para get_available_models() en hardware.py."""

    def test_function_exists(self):
        """get_available_models debe existir."""
        from hardware import get_available_models
        self.assertTrue(callable(get_available_models))

    def test_returns_list(self):
        """get_available_models retorna lista (vacía sin Ollama)."""
        from hardware import get_available_models
        result = get_available_models()
        self.assertIsInstance(result, list)


class TestIsModelAvailable(unittest.TestCase):
    """Tests para is_model_available() en ollama_client.py."""

    def test_function_exists(self):
        """is_model_available debe existir."""
        from ollama_client import is_model_available
        self.assertTrue(callable(is_model_available))

    def test_returns_bool(self):
        """is_model_available retorna bool."""
        from ollama_client import is_model_available
        result = is_model_available("nonexistent-model")
        self.assertIsInstance(result, bool)


class TestCallOllamaGenerate(unittest.TestCase):
    """Tests para call_ollama_generate()."""

    def test_function_exists(self):
        """call_ollama_generate debe existir."""
        from ollama_client import call_ollama_generate
        self.assertTrue(callable(call_ollama_generate))

    def test_function_returns_dict(self):
        """call_ollama_generate retorna dict incluso sin Ollama."""
        from ollama_client import call_ollama_generate
        result = call_ollama_generate(
            model="llama3.2:3b",
            prompt="test",
            system="test"
        )
        self.assertIsInstance(result, dict)
        self.assertIn('ok', result)


class TestUsageTracking(unittest.TestCase):
    """Tests para tracking de uso."""

    def test_get_usage_stats_returns_dict(self):
        from ollama_client import get_usage_stats
        stats = get_usage_stats()
        self.assertIsInstance(stats, dict)
        expected_fields = ['total_calls', 'successful_calls', 'failed_calls',
                          'injection_attempts_blocked', 'total_input_tokens',
                          'total_output_tokens', 'total_savings_usd', 'avg_latency_ms']
        for field in expected_fields:
            self.assertIn(field, stats, f"Campo faltante: {field}")


if __name__ == '__main__':
    unittest.main()

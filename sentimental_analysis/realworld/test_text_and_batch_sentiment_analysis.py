import unittest
from unittest.mock import patch

import pytest
# import os, sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# from views import detailed_analysis_sentence
from realworld.views import detect_language, analyze_sentiment, textanalysis, batch_analysis
from django.test import RequestFactory

from langdetect import DetectorFactory
DetectorFactory.seed = 0

class SentimentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    # Test cases for detect_language
    def test_detect_language_english(self):
        self.assertEqual(detect_language(["Hello, how are you?"]), "en")

    def test_detect_language_spanish(self):
        self.assertEqual(detect_language(["Hola, cómo estás?"]), "es")

    def test_detect_language_french(self):
        self.assertEqual(detect_language(["Bonjour, comment ça va?"]), "fr")

    @pytest.mark.xfail # langdetect known to be non-deterministic - on kurt's machine, somehow this is Somali?
    def test_detect_language_mixed(self):
        self.assertEqual(detect_language(["How are you? Are you okay? bonjour!"]), "en")  # Most common language

    def test_detect_language_unknown(self):
        self.assertEqual(detect_language([""]), "unknown")

    def test_detect_language_unknown(self):
        self.assertEqual(detect_language(["1235.908"]), "unknown")

    # Test cases for analyze_sentiment
    @pytest.mark.xfail # non-deteministic?
    @patch("views.classifiers.SpanishClassifier.predict")
    def test_analyze_sentiment_spanish(self, mock_predict):
        mock_predict.return_value = {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
        result = analyze_sentiment("Este es un gran día", "es")
        self.assertEqual(result, {'pos': 0.8, 'neu': 0.1, 'neg': 0.1})

    @pytest.mark.xfail # non-deteministic?
    def test_analyze_sentiment_english_translation(self):
        result = analyze_sentiment("C'est une belle journée", "fr")
        self.assertEqual(result, {"pos": 0.583, "neu":0.417, "neg": 0.0})

    # Test cases for textanalysis
    def test_textanalysis_post(self):
        request = self.factory.post("/", {"textField": "Hello, this is a test."})
        response = textanalysis(request)
        self.assertEqual(response.status_code, 200)

    def test_textanalysis_get(self):
        request = self.factory.get("/")
        response = textanalysis(request)
        self.assertEqual(response.status_code, 200)

    # Test cases for batch_analysis
    def test_batch_analysis_post(self):
        request = self.factory.post("/", {"batchTextField": "Hello world.\nThis is a test."})
        response = batch_analysis(request)
        self.assertEqual(response.status_code, 200)

    def test_batch_analysis_get(self):
        request = self.factory.get("/")
        response = batch_analysis(request)
        self.assertEqual(response.status_code, 200)

    # Additional tests for edge cases
    def test_detect_language_numbers(self):
        self.assertEqual(detect_language(["123456"]), "unknown")

    def test_detect_language_multiple_sentences(self):
        self.assertEqual(detect_language(["Hola. Adiós. Hello."]), "es")  # Most frequent

    def test_analyze_sentiment_empty_text(self):
        result = analyze_sentiment("", "en")
        self.assertEqual(result, {"pos": 0.0, "neu": 0.0, "neg": 0.0})

    def test_textanalysis_punctuation(self):
        request = self.factory.post("/", {"textField": "!!! ??? ..."})
        response = textanalysis(request)
        self.assertEqual(response.status_code, 200)

    def test_detect_language_special_characters(self):
        self.assertEqual(detect_language(["@#$%^&*()"]), "unknown")

    def test_batch_analysis_empty_input(self):
        request = self.factory.post("/", {"batchTextField": ""})
        response = batch_analysis(request)
        self.assertEqual(response.status_code, 200)

    def test_batch_analysis_large_texts(self):
        large_text = "This is a long text. " * 100
        request = self.factory.post("/", {"batchTextField": large_text})
        response = batch_analysis(request)
        self.assertEqual(response.status_code, 200)

    def test_batch_analysis_mixed_languages(self):
        request = self.factory.post("/", {"batchTextField": "Hola mundo.\nBonjour le monde.\nHello world."})
        response = batch_analysis(request)
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()

import unittest
from historical_text_classifier_xgboost.xgboost_model import XGBoostTextClassifier

class TestXGBoostModel(unittest.TestCase):
    def setUp(self):
        self.model = XGBoostTextClassifier()
        self.model.train()

    def test_predict(self):
        prediction = self.model.predict("В 1757 г. для размещения Университета, основанного М.В. Ломоносовым в 1755 г., была приобретена усадьба князя Репнина на Моховой.")
        self.assertIn(prediction, [0, 1])

    def test_extract_valuable_passages(self):
        passages = self.model.extract_valuable_passages("В 1757 г. для размещения Университета, основанного М.В. Ломоносовым в 1755 г., была приобретена усадьба князя Репнина на Моховой.\nСегодня солнечная погода, и я гулял по парку. Это было замечательно.\nЭрмитаж — крупнейший музей мира, расположенный в Санкт-Петербурге.")
        self.assertTrue(isinstance(passages, list))
        for passage, score in passages:
            self.assertTrue(isinstance(passage, str))
            self.assertTrue(isinstance(score, float))
            self.assertGreaterEqual(score, 0.5)

if __name__ == '__main__':
    unittest.main()
from unittest import TestCase
from src.data_processing import *
from test_utils import *


class DataProcessingTestCase(TestCase):
    def test_impute(self):

        data = get_mock_data()

        # Impute 0 of q_OpeningHours
        data_imputed = impute(data)

        self.assertEqual(data_imputed["q_OpeningHours"].mean(), 10)

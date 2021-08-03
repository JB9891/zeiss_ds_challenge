import sys


from unittest import TestCase
import pandas as pd
from src.data_processing import *
from test_utils import *


class DataProcessingTestCase(TestCase):

    def test_impute(self):

        data = get_mock_data()

        # Impute 0 of q_OpeningHours
        data_imputed = impute(data)

        pass

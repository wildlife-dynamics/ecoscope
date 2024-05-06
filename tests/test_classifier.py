import pandas as pd

import ecoscope


class TestClassifier(object):
    def setup_method(self):
        self.df = pd.DataFrame(
            {
                "column": [1, 2, 3, 4, 5],
            }
        )

    def test_apply_classification(self):
        result = ecoscope.analysis.apply_classification(self.df, 2, cls_method="natural_breaks")
        assert result == [1, 2, 5]

import pytest
import numpy as np
from ir_eval_numba.metrics import (
    recall,
    precision,
    average_precision,
    mean_average_precision,
    ndcg,
    reciprocal_rank,
    mean_reciprocal_rank,
)

# Sample data generated with:
# total_count_items = 100
# total_relevant_items = 25
# rng = np.random.default_rng()
# actual = rng.choice(total_count_items, total_relevant_items, replace=False)

actual = np.array([ 4, 79, 32, 45, 14, 46, 53, 15,  3, 54, 68, 99, 75, 82, 35, 27, 73,
    20, 25, 66, 11, 58, 31,  8, 85])
predicted = np.array([1, 2, 62, 84, 3, 4, 81, 14, 5, 67])
# intersection: {3, 4, 14}

class TestRecall:
  def test_recall_k_5(self):
    result = recall(actual, predicted, 5)
    assert result == pytest.approx(0.04) # 1 out of 25
  
  def test_recall_k_10(self):
    result = recall(actual, predicted, 10)
    assert result == pytest.approx(0.12) # 3 out of 25

class TestPrecision:
  def test_precision_k_5(self):
    result = precision(actual, predicted, 5)
    assert result == pytest.approx(0.2) # 1 out of 5
  
  def test_precision_k_10(self):
    result = precision(actual, predicted, 10)
    assert result == pytest.approx(0.3) # 3 out of 10

class TestAveragePrecision:
  def test_average_precision_basic(self):
    # basic inputs
    result = average_precision(np.array([1,3,5]), np.array([1,2,3,4,5]), 5)
    assert result == pytest.approx(0.7555555555555555) # (1 + 0.67 + 0.6) / 3 = 0.75555

  def test_precision_k_5(self):
    result = average_precision(actual, predicted, 5)
    assert result == pytest.approx(0.2)

  def test_precision_k_10(self):
    result = average_precision(actual, predicted, 10)
    assert result == pytest.approx(0.30277777777777776)


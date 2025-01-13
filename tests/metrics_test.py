import random
import pytest
import numpy as np
from numba.typed import List
from ir_eval_numba.metrics import (
    recall,
    precision,
    f1_score,
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
  def test_is_numba_func(self):
    assert hasattr(recall, '__numba__')

  def test_recall_zero(self):
    result = recall(actual, np.array([5,6,7,9,10]), 5)
    assert result == pytest.approx(0.0) # 0 out of 25

  def test_recall_perfect(self):
    actual_cloned_list = list(actual)
    random.shuffle(actual_cloned_list) # does in place shuffle
    actual_cloned = np.array(actual_cloned_list)
    result = recall(actual, actual_cloned, len(actual))
    assert result == pytest.approx(1.0) # 5 out of 5

  def test_recall_k_5(self):
    result = recall(actual, predicted, 5)
    assert result == pytest.approx(0.04) # 1 out of 25
  
  def test_recall_k_10(self):
    result = recall(actual, predicted, 10)
    assert result == pytest.approx(0.12) # 3 out of 25

class TestPrecision:
  def test_is_numba_func(self):
    assert hasattr(precision, '__numba__')

  def test_precision_zero(self):
    result = precision(actual, np.array([5,6,7,9,10]), 5)
    assert result == pytest.approx(0.0) # 0 out of 25

  def test_precision_k_5_perfect(self):
    result = precision(actual, np.array([11, 58, 31, 8, 85]), 5)
    assert result == pytest.approx(1.0) # 5 out of 5

  def test_precision_k_5(self):
    result = precision(actual, predicted, 5)
    assert result == pytest.approx(0.2) # 1 out of 5
  
  def test_precision_k_10(self):
    result = precision(actual, predicted, 10)
    assert result == pytest.approx(0.3) # 3 out of 10

class TestF1:
  def test_is_numba_func(self):
    assert hasattr(f1_score, '__numba__')

  def test_f1_zero(self):
    result = f1_score(actual, np.array([5,6,7,9,10]), 5)
    assert result == pytest.approx(0.0)

  def test_f1_perfect(self):
    result = f1_score(np.array([8, 11, 31, 58, 85]), np.array([11, 58, 31, 8, 85]), 5)
    assert result == pytest.approx(1.0)

  def test_f1_k_5(self):
    result = f1_score(actual, predicted, 5)
    assert result == pytest.approx(0.06666666666666667) # precision: 0.2, recall: 0.04
  
  def test_f1_k_10(self):
    result = f1_score(actual, predicted, 10)
    assert result == pytest.approx(0.17142857142857143) # precision: 0.3, recall: 0.12

class TestAveragePrecision:
  def test_is_numba_func(self):
    assert hasattr(average_precision, '__numba__')

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

class TestMeanAveragePrecision:
  def test_is_numba_func(self):
    assert hasattr(mean_average_precision, '__numba__')

  def test_mean_average_precision_basic(self):
    # basic inputs
    actual_list = List([
      np.array([1,3,5,10]),
      np.array([2,4,6,8]),
      np.array([7,8,9])
    ]) # type: ignore

    predicted_list = List([
      np.array([1,2,3,4,5]),
      np.array([9,2,3,1,5]),
      np.array([4,5,9,8,3])
    ]) # type: ignore
    result = mean_average_precision(actual_list, predicted_list, 5)
    # ap values: [0.7555555555555555, 0.5, 0.41666666666666663]
    assert result == pytest.approx(0.5574074074074074)

  def test_mean_average_precision_pydoc(self):
    # example from pydoc string
    actual_list = List([np.array([1, 2, 3]), np.array([2, 3, 4])]) # type: ignore

    predicted_list = List([np.array([1, 4, 2, 3]), np.array([2, 3, 5, 4])]) # type: ignore
    result = mean_average_precision(actual_list, predicted_list, 3)
    # ap values: [0.8333333333333333, 1.0]
    assert result == pytest.approx(0.9166666666666666)

class TestNCDG:
  def test_is_numba_func(self):
    assert hasattr(ndcg, '__numba__')

  def test_ndcg_k_5(self):
    result = ndcg(actual, predicted, 5)
    assert result == pytest.approx(0.13120507751234178)
  
  def test_ndcg_k_10(self):
    result = ndcg(actual, predicted, 10)
    assert result == pytest.approx(0.23297260855707355)

class TestReciprocalRank:
  def test_is_numba_func(self):
    assert hasattr(reciprocal_rank, '__numba__')

  def test_reciprocal_rank_k_10(self):
    result = reciprocal_rank(actual, predicted, 10)
    assert result == pytest.approx(0.2) # found at position 5
  
  def test_reciprocal_rank_zero(self):
    result = reciprocal_rank(np.array([1,2,3]), np.array([4,5,6,7,8]), 5)
    assert result == pytest.approx(0) # no relevant items found

class TestMeanReciprocalRank:
  def test_is_numba_func(self):
    assert hasattr(mean_reciprocal_rank, '__numba__')

  def test_mean_reciprocal_rank_basic(self):
    # basic inputs
    actual_list = List([
      np.array([1,3,5]),
      np.array([2,4,6]),
      np.array([7,8,9]),
      np.array([7,8,9])
    ]) # type: ignore

    predicted_list = List([
      np.array([1,2,3,4,5]),
      np.array([9,2,3,1,5]),
      np.array([4,5,9,8,3]),
      np.array([1,2,3,4,5])
    ]) # type: ignore
    result = mean_reciprocal_rank(actual_list, predicted_list, 5)
    # rr values: [1.0, 0.5, 0.333, 0]
    assert result == pytest.approx(0.4583333333333333)
  
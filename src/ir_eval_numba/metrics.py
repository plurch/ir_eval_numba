import math
import numpy as np
import numpy.typing as npt
from numba import njit
from typing import TypeVar

IntType = TypeVar('IntType', np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)

@njit(nogil=True, cache=True)
def find_relevant_indices(actual: npt.NDArray[IntType], predicted: npt.NDArray[IntType], k: int) -> npt.NDArray[IntType]:
  """
  Find indices of top-k predictions that are relevant items

  numba does not support np.isin(), so use this implementation

  Args:
    actual (npt.NDArray[IntType]): An array of ground truth relevant items.
    predicted (npt.NDArray[IntType]): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    npt.NDArray[IntType]: The mean of all values in the array
  """
  actual_set = set(actual)
  idxs = []

  for i in range(k):
    if predicted[i] in actual_set:
      idxs.append(i)

  return np.array(idxs)

@njit(nogil=True, cache=True)
def mean_of_array(l: npt.NDArray[IntType]) -> float:
  """
  Calculate the mean of all values in an array

  Args:
    l (npt.NDArray[IntType]): An array of float values

  Returns:
    float: The mean of all values in the array
  """
  return sum(l) / len(l)

@njit(nogil=True, cache=True)
def recall(actual: npt.NDArray[IntType], predicted: npt.NDArray[IntType], k: int) -> float:
  """
  Calculate the recall@k metric.

  Recall is defined as the ratio of the total number of relevant items retrieved within the top-k predictions to the total number of relevant items in the entire database.

  Recall =  Total number of items retrieved that are relevant/Total number of relevant items in the database.

  Args:
    actual (npt.NDArray[IntType]): An array of ground truth relevant items.
    predicted (npt.NDArray[IntType]): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    float: The recall value at rank k, ranging from 0 to 1.
           A value of 1 indicates perfect recall, while 0 indicates no relevant items retrieved.

  Notes:
    - This function assumes the `predicted` array is sorted in descending order of relevance.
    - If k is larger than the length of the `predicted` array, it will consider the entire array.
  """
  actual_set = set(actual)
  top_k_predictions = set(predicted[:k])
  count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
  return count_relevant_in_top_k / float(len(actual_set))

@njit(nogil=True, cache=True)
def precision(actual: npt.NDArray[IntType], predicted: npt.NDArray[IntType], k: int) -> float:
  """
  Calculate the precision@k metric.

  Precision is defined as the ratio of the total number of relevant items retrieved
  within the top-k predictions to the total number of returned items (k).

  Precision =  Total number of items retrieved that are relevant/Total number of items that are retrieved.

  Args:
    actual (npt.NDArray[IntType]): An array of ground truth relevant items.
    predicted (npt.NDArray[IntType]): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    float: The precision value at rank k, ranging from 0 to 1.
           A value of 1 indicates perfect precision, while 0 indicates no relevant items retrieved.

  Notes:
    - This function assumes the `predicted` array is sorted in descending order of relevance.
    - If k is larger than the length of the `predicted` array, it will consider the entire array.
  """
  actual_set = set(actual)
  top_k_predictions = set(predicted[:k])
  count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
  return count_relevant_in_top_k / float(k)

@njit(nogil=True, cache=True)
def f1_score(actual: npt.NDArray[IntType], predicted: npt.NDArray[IntType], k: int) -> float:
  """
  Calculate the F1-score @k metric.

  The F1-score is calculated as the harmonic mean of precision and recall. The formula is:
  F1 = 2 * (Precision * Recall) / (Precision + Recall)

  The F1 score provides a balanced view of a system's performance by taking into account both precision and recall. This is especially important when evaluating information retrieval systems, where finding all relevant documents is just as important as minimizing irrelevant ones.

  Args:
    actual (npt.NDArray[IntType]): An array of ground truth relevant items.
    predicted (npt.NDArray[IntType]): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    float: The F1 score value at rank k, ranging from 0 to 1.
           A value of 1 indicates perfect precision and recall, while 0 indicates either precision or recall is zero.
  """
  recall_score = recall(actual, predicted, k)
  precision_score = precision(actual, predicted, k)

  if recall_score == 0  or precision_score == 0:
    return float(0)

  return 2 * (recall_score * precision_score) / (recall_score + precision_score)

@njit(nogil=True, cache=True)
def average_precision(actual: npt.NDArray[IntType], predicted: npt.NDArray[IntType], k: int) -> float:
  """
  Computes the Average Precision (AP) at a specified rank `k`.

  Average Precision (AP) is a metric used to evaluate the relevance of predicted rankings 
  in information retrieval tasks. It is calculated as the mean of precision values at 
  each rank where a relevant item is retrieved within the top `k` predictions.

  Args:
      actual (npt.NDArray[IntType]): A list of integers representing the ground truth relevant items.
      predicted (npt.NDArray[IntType]): A list of integers representing the predicted rankings of items.
      k (int): The maximum number of top-ranked items to consider for evaluation.

  Returns:
      float: The Average Precision score. If no relevant items are retrieved within the
      top `k` predictions, the function may raise a division by zero error or return `NaN`.
  """
  relevant_idxs = find_relevant_indices(actual, predicted, k)
  precision_array = np.array([precision(actual, predicted, i+1) for i in relevant_idxs])
  return mean_of_array(precision_array)

@njit(nogil=True, cache=True)
def mean_average_precision(actual_list: list[npt.NDArray[IntType]], predicted_list: list[npt.NDArray[IntType]], k: int) -> float:
  """
  Computes the Mean Average Precision (MAP) at a specified rank `k`.

  It is the mean of the Average Precision (AP) scores computed for multiple 
  queries.

  Args:
      actual_list (numba.typed.List[npt.NDArray[IntType]]): A list of arrays where each inner list represents 
          the ground truth relevant items for a query
      predicted_list (numba.typed.List[npt.NDArray[IntType]]): A list of arrays where each inner list represents 
          the predicted rankings of items for a query
      k (int): The maximum number of top-ranked items to consider for each prediction.

  Returns:
      float: The Mean Average Precision score, which is the mean of AP scores across all 
      queries.

  Raises:
      AssertionError: If the lengths of `actual_list` and `predicted_list` are not equal.
  """
  assert len(actual_list) == len(predicted_list)

  ap_values = np.array([average_precision(actual_list[i], predicted_list[i], k) for i in range(len(actual_list))])
  return mean_of_array(ap_values) 

@njit(nogil=True, cache=True)
def ndcg(actual: npt.NDArray[IntType], predicted: npt.NDArray[IntType], k: int) -> float:
  """
  Computes the Normalized Discounted Cumulative Gain (nDCG) at a specified rank `k`.

  nDCG evaluates the quality of a predicted ranking by comparing it to an ideal ranking 
  (i.e., perfect ordering of relevant items). It accounts for the position of relevant 
  items in the ranking, giving higher weight to items appearing earlier.

  Args:
      actual (npt.NDArray[IntType]): A list of integers representing the ground truth relevant items.
      predicted (npt.NDArray[IntType]): A list of integers representing the predicted rankings of items.
      k (int): The maximum number of top-ranked items to consider for evaluation.

  Returns:
      float: The nDCG score, which ranges from 0 to 1. A value of 1 indicates a perfect 
      ranking. Returns 0 if there are no relevant items in the top `k` predictions or if 
      the ideal DCG (iDCG) is zero.
  """
  actual_set = set(actual)

  # discounted cumulative gain
  # `i+2` due to zero indexing
  dcg = sum([1.0/math.log2(i+2) for i in range(k) if predicted[i] in actual_set])
  # ideal discounted cumulative gain (ie. perfect results returned)
  idcg = sum([1.0/math.log2(i+2) for i in range(min(k, len(actual_set)))])
  return dcg / idcg

@njit(nogil=True, cache=True)
def reciprocal_rank(actual: npt.NDArray[IntType], predicted: npt.NDArray[IntType], k: int) -> float:
  """
  Computes the Reciprocal Rank (RR) at a specified rank `k`.

  Reciprocal Rank (RR) assigns a score based on the reciprocal of the rank at which the first relevant item is found.

  Args:
      actual (npt.NDArray[IntType]): A list of integers representing the ground truth relevant items.
      predicted (npt.NDArray[IntType]): A list of integers representing the predicted rankings of items.
      k (int): The maximum number of top-ranked items to consider for evaluation.

  Returns:
      float: The Reciprocal Rank score. A value of 0 is returned if no relevant items are 
      found within the top `k` predictions. Otherwise, the score is `1 / rank`, where 
      `rank` is the position (1-based) of the first relevant item.

  Notes:
      - The function assumes zero-based indexing for the `predicted` list.
      - If `k` exceeds the length of `predicted`, only the available elements in `predicted` 
        are considered.
      - This metric focuses only on the rank of the first relevant item, ignoring others.
  """
  actual_set = set(actual)

  for i in range(k):
    if predicted[i] in actual_set:
      return 1 / float(i + 1)
  
  return float(0)

@njit(nogil=True, cache=True)
def mean_reciprocal_rank(actual_list: list[npt.NDArray[IntType]], predicted_list: list[npt.NDArray[IntType]], k: int) -> float:
  """
  Computes the Mean Reciprocal Rank (MRR) at a specified rank `k`.

  It calculates the mean of the Reciprocal Rank (RR) scores for a set of queries.

  Args:
      actual_list (numba.typed.List[npt.NDArray[IntType]]): A list of arrays where each inner list represents the 
          ground truth relevant items for a query or task.
      predicted_list (numba.typed.List[npt.NDArray[IntType]]): A list of arrays where each inner list represents 
          the predicted rankings of items for a query or task.
      k (int): The maximum number of top-ranked items to consider for each prediction.

  Returns:
      float: The Mean Reciprocal Rank score, which is the average of RR scores across all 
      queries. Returns 0 if there are no queries or no relevant items in any predictions.

  Raises:
      AssertionError: If the lengths of `actual_list` and `predicted_list` are not equal.
  """

  assert len(actual_list) == len(predicted_list)

  rr_values = np.array([reciprocal_rank(actual_list[i], predicted_list[i], k) for i in range(len(actual_list))])
  return mean_of_array(rr_values)

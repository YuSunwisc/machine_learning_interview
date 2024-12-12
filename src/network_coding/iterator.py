# follow up 如何优化 memory
# 传入的参数是 iterator of iterator, 不需要 list of iterator，节省内存

# Problem 1: Interface
import unittest
from abc import ABC, abstractmethod

class ResumableIterator(ABC):
    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

    @abstractmethod
    def getState(self):
        pass

    @abstractmethod
    def setState(self):
        pass

# Problem 2: ResumableListIterator

class ResumableListIterator(ResumableIterator):
    def __init__(self, data):
        if not isinstance(data, list):
            raise TypeError("Input must be a list.")

        self.data = data
        self.index = 0

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        res = self.data[self.index]
        self.index += 1
        return res

    def getState(self):
        return self.index

    def setState(self, state):
        if not isinstance(state, int):
            raise TypeError("Input 'state' must be a integer.")
        
        if not (0 <= state <= len(self.data)):
            raise ValueError(f"Input 'state' must be a integer between 0 and {len(self.data)}.")
        
        self.index = state

class TestResumableListIterator(unittest.TestCase):
    
    # 1. init
    def test_valid_input(self):
        data = (7,8,9)

        with self.assertRaises(TypeError):
            test_iter = ResumableListIterator(data)

    def test_empty_input(self):
        data = []
        test_iter = ResumableListIterator(data)

        with self.assertRaises(StopIteration):
            next(test_iter)
            
        
    # 2. next
    def test_valid_next(self):
        data = [7,8,9]
        test_iter = ResumableListIterator(data)

        self.assertEqual(next(test_iter), 7)
        self.assertEqual(next(test_iter), 8)
        self.assertEqual(next(test_iter), 9)

        with self.assertRaises(StopIteration):
            next(test_iter)
            
    # 3. getState
    def test_valid_getState(self):
        data = [7,8,9]
        test_iter = ResumableListIterator(data)

        self.assertEqual(test_iter.getState(), 0)
        next(test_iter)
        self.assertEqual(test_iter.getState(), 1)
        next(test_iter)
        next(test_iter)
        self.assertEqual(test_iter.getState(), 3)

        with self.assertRaises(StopIteration):
            next(test_iter)

        self.assertEqual(test_iter.getState(), 3)

    # 4. setState
    def test_valid_setState(self):

        data = [7,8,9]
        test_iter = ResumableListIterator(data)

        with self.assertRaises(TypeError):
            test_iter.setState('1')
        with self.assertRaises(ValueError):
            test_iter.setState(-1)
        with self.assertRaises(ValueError):
            test_iter.setState(4)

        next(test_iter)
        state = test_iter.getState()
        next(test_iter)
        next(test_iter)
        test_iter.setState(state)

        self.assertEqual(test_iter.getState(), 1)
        
    # 5. iter
    def test_valid_iter(self):
        data = [7,8,9]
        test_iter = ResumableListIterator(data)
        
        self.assertIs(iter(test_iter), test_iter)


# Problem 3: ResumableMultiFileIterator

class ResumableMultiFileIterator(ResumableIterator):
  def __init__(self, iterators):
    if not isinstance(iterators, list) or not all(isinstance(it, ResumableListIterator) for it in iterators):
      raise ValueError("Input must be a list of ResumableListIterator!")

    self.iterators = iterators
    self.current_index = 0
    self.initial_states = [it.getState() for it in iterators]

  def __next__(self):
    while self.current_index < len(self.iterators):
      cur_iterator = self.iterators[self.current_index]
      try:
        return next(cur_iterator)
      except StopIteration:
        cur_iterator.setState(self.initial_states[self.current_index])
        self.current_index += 1

    raise StopIteration

  def getState(self):
    return {
        "current_index": self.current_index,
        "states": [it.getState() for it in self.iterators]
    }

  def setState(self, state):
    ## all valid input check

    if not isinstance(state, dict):
      raise TypeError("Input must be a dictionary!")
    if not "current_index" in state or not "states" in state:
      raise ValueError("Input must contains 'current_index' and 'states'!")
    if not isinstance(state["states"], list) or not isinstance(state["current_index"], int) \
    or not all(isinstance(ele, int) for ele in state['states']):
      raise TypeError("'current_index' must be integer and 'states' must be a list of intergers!")
    if not (0 <= state['current_index'] <= len(self.iterators)):
      raise ValueError(f"'current_index' must be from 0 to {len(self.iterators)}!")

    ## update States
    if len(state['states']) != len(self.iterators):
      raise ValueError("Number of state['states'] must be equal to the length of the input!")

    self.current_index = state['current_index']
    for i, iterator in enumerate(self.iterators):
      iterator.setState(state['states'][i])


class TestResumableMultiFileIterator(unittest.TestCase):
    def setUp(self):
        # Initialize ResumableListIterators
        self.iterator1 = ResumableListIterator([1, 2, 3])
        self.iterator2 = ResumableListIterator([])
        self.iterator3 = ResumableListIterator([])
        self.iterator4 = ResumableListIterator([4, 5, 6, 7])

        # Set initial states
        self.iterator1.setState(1)
        self.iterator2.setState(0)
        self.iterator3.setState(0)
        self.iterator4.setState(2)

        # Create ResumableMultiFileIterator
        self.multi_iterator = ResumableMultiFileIterator(
            [self.iterator1,
             self.iterator2,
             self.iterator3,
             self.iterator4]
        )

    def test_iteration(self):
        # Test iteration over the ResumableMultiFileIterator
        self.assertEqual(next(self.multi_iterator), 2)  # From iterator1 (initial state = 1)
        self.assertEqual(next(self.multi_iterator), 3)  # From iterator1
        self.assertEqual(next(self.multi_iterator), 6)  # From iterator4 (initial state = 2)
        self.assertEqual(next(self.multi_iterator), 7)  # From iterator4
        with self.assertRaises(StopIteration):
            next(self.multi_iterator)  # All iterators exhausted

    def test_reset_after_iteration(self):

        # Verify that all iterators have been reset to their initial states
        self.assertEqual(self.iterator1.getState(), 1)
        self.assertEqual(self.iterator2.getState(), 0)
        self.assertEqual(self.iterator3.getState(), 0)
        self.assertEqual(self.iterator4.getState(), 2)

    def test_get_and_set_state(self):
        # Partially iterate through the multi_iterator
        self.assertEqual(next(self.multi_iterator), 2)  # From iterator1
        self.assertEqual(next(self.multi_iterator), 3)  # From iterator1

        # Save the current state
        state = self.multi_iterator.getState()

        # Continue iteration
        self.assertEqual(next(self.multi_iterator), 6)  # From iterator4
        self.assertEqual(next(self.multi_iterator), 7)  # From iterator4

        # Restore state
        self.multi_iterator.setState(state)
        self.assertEqual(next(self.multi_iterator), 6)  # From iterator4
        self.assertEqual(next(self.multi_iterator), 7)  # From iterator4

        with self.assertRaises(StopIteration):
            next(self.multi_iterator)

    def test_valid_setState(self):
      # TypeErrors
        with self.assertRaises(TypeError):
          state = [
              0,
              [3, 0, 0, 2]
          ]
          self.multi_iterator.setState(state)

        with self.assertRaises(TypeError):
          state = {
              "current_index": '0',
              "states": [3, 0, 0, 2]
          }
          self.multi_iterator.setState(state)

        with self.assertRaises(TypeError):
          state = {
              "current_index": 0,
              "states": (3, 0, 0, 2)
          }
          self.multi_iterator.setState(state)

        with self.assertRaises(TypeError):
          state = {
              "current_index": 0,
              "states": ['3', 0, 0, 2]
          }
          self.multi_iterator.setState(state)

        # ValueErrors

        with self.assertRaises(ValueError):
          state = {
              "current_index": 0,
          }
          self.multi_iterator.setState(state)

        with self.assertRaises(ValueError):
          state = {
              "states": [3, 0, 0, 2]
          }
          self.multi_iterator.setState(state)

        with self.assertRaises(ValueError):
          state = {
              "current_index": 6,
              "states": [3, 0, 0, 2]
          }
          self.multi_iterator.setState(state)


        with self.assertRaises(ValueError):
          state = {
              "current_index": 0,
              "states": [-1, 0, 0, 2]
          }
          self.multi_iterator.setState(state)


    def test_valid_iter(self):
      self.assertIs(iter(self.multi_iterator), self.multi_iterator)


# Problem 4: EfficientResumableMultiFileIterator

class ResumableIteratorOfMultiFileIterator(ResumableIterator):
    def __init__(self, data_lists):
        self.data_lists = data_lists
        self.iterators = [ResumableListIterator(data) for data in data_lists]
        self.index = 0  # Index of the current iterator

    def __next__(self):

        if self.index >= len(self.iterators):
            raise StopIteration
        result = self.iterators[self.index]
        self.index += 1
        return result

    def getState(self):
        return {"index": self.index}

    def setState(self, state):

        if "index" not in state or not isinstance(state["index"], int):
            raise ValueError("State must include 'index' as an integer.")
        if not (0 <= state["index"] <= len(self.iterators)):
            raise ValueError(f"Index must be between 0 and {len(self.iterators)}.")
        self.index = state["index"]


class EfficientResumableMultiFileIterator(ResumableIterator):
    def __init__(self, iterator_data):
        """
        Initialize EfficientResumableMultiFileIterator.

        Args:
            iterator_data (ResumableIterator): A resumable iterator, where each
                                               element is a ResumableListIterator.
        """
        if not isinstance(iterator_data, ResumableIterator):
            raise TypeError("Input must be a ResumableIterator!")
        self.outer_iter = iterator_data  # The outer iterator
        self.outer_index = 0            # Current index of the outer iterator
        self.inner_iter = None          # Current inner iterator
        self.inner_initial_index = None  # Initial index of the current inner iterator

    def __next__(self):
        """
        Fetch the next element from the current inner iterator.
        If the inner iterator is exhausted, reset it and move to the next outer iterator.
        """
        while True:
            if self.inner_iter is None:
                try:
                    self.inner_iter = next(self.outer_iter)
                    self.inner_initial_index = self.inner_iter.getState()
                except StopIteration:
                    # Outer iterator exhausted
                    raise StopIteration("No more elements in EfficientResumableMultiFileIterator.")
            
            try:
                return next(self.inner_iter)
            except StopIteration:
                # Inner iterator exhausted, reset and move to next
                self.inner_iter.setState(self.inner_initial_index)
                self.inner_iter = None
                self.outer_index += 1

    def getState(self):
        """
        Get the current state of the iterator.

        Returns:
            dict: A dictionary containing the outer and inner state.
        """
        if self.inner_iter is None:
            return {"outer_index": self.outer_index, "inner_state": None}
        return {"outer_index": self.outer_index, "inner_state": self.inner_iter.getState()}

    def setState(self, state):
        """
        Restore the iterator to a previous state.

        Args:
            state (dict): A dictionary containing the state to restore.
        """
        if not isinstance(state, dict):
            raise TypeError("State must be a dictionary!")
        if "outer_index" not in state or "inner_state" not in state:
            raise ValueError("State must include 'outer_index' and 'inner_state'!")

        outer_index = state["outer_index"]
        inner_state = state["inner_state"]

        # Reset outer iterator to the start
        self.outer_iter.setState({"index": outer_index})
        self.outer_index = outer_index
        self.inner_iter = None

        # Load the correct inner iterator
        try:
            self.inner_iter = next(self.outer_iter)
        except StopIteration:
            self.inner_iter = None

        # Restore the inner iterator state
        if self.inner_iter and inner_state is not None:
            self.inner_iter.setState(inner_state)


class TestEfficientResumableMultiFileIterator(unittest.TestCase):
    def setUp(self):
        self.data_lists = [
            [1, 2, 3],
            [],
            [4],
            [5, 6, 7]
        ]
        self.outer_iterator = ResumableIteratorOfMultiFileIterator(self.data_lists)
        self.efficient_iterator = EfficientResumableMultiFileIterator(self.outer_iterator)

    def test_iteration(self):
        results = []
        try:
            while True:
                results.append(next(self.efficient_iterator))
        except StopIteration:
            pass
        self.assertEqual(results, [1, 2, 3, 4, 5, 6, 7])

    def test_state_management(self):
        # Iterate partially
        self.assertEqual(next(self.efficient_iterator), 1)
        self.assertEqual(next(self.efficient_iterator), 2)
        state = self.efficient_iterator.getState()

        # Continue iteration
        results = []
        try:
            while True:
                results.append(next(self.efficient_iterator))
        except StopIteration:
            pass

        self.assertEqual(results, [3, 4, 5, 6, 7])

        # Restore state
        self.efficient_iterator.setState(state)
        restored_results = []
        try:
            while True:
                restored_results.append(next(self.efficient_iterator))
        except StopIteration:
            pass

        self.assertEqual(restored_results, [3, 4, 5, 6, 7])

if __name__ == "__main__":
    unittest.main()




import unittest

from c_list_iterator import ListIterator
from d_multi_file_iterator import MultiFileIterator
from e_efficient_multi_file_iterator import EfficientMultiFileIterator
from jsonl_file_iterator import JsonlFileIterator

##############################################
# PART 1: Implement general purpose test_resumable_iter
# that iterates over the inputted list and at each point:
# (1) checks the value is what's expected and
# (2) saves a state and later verifies that resuming 
# from the state matches same output.
# Usage:
#.  expected = ["o", "p", "e", "n"]
#.  it = ResumableIterator(expected)
#.  test_resumable_iter(it, expected) <--
##############################################


def test_resumable_iter(it, expected):

    initial_s = it.get_state()
    for i in range(len(expected)):
      assert next(it) == expected[i]

    try:
      next(it)
    except StopIteration:
      assert True

    it.set_state(initial_s)

    
    


    for i in range(len(expected)):
      initial_state = it.get_state()

      prev_state = it.get_state()

      for j in range(i, len(expected)):

        it.set_state(prev_state)
        assert next(it) == expected[j]
        prev_state = it.get_state()
      try:
        next(it)
      except StopIteration:
        assert True

      it.set_state(initial_state)
      next(it)


    
          



class TestResumableIterator(unittest.TestCase):

  def test_sanity(self):
    assert True

  ##############################################
  # PART 2: test list resumable iterator
  ##############################################
  def test_list_1(self):
    it = ListIterator([1,2,3])
    assert next(it) == 1
    s = it.get_state()
    assert next(it) == 2
    assert next(it) == 3
    it.set_state(s)  # go back to previous point of iteration!
    assert next(it) == 2
    assert next(it) == 3

    print("test_list_1 passed!")

  def test_list_2(self):
    OPENAI = ["o", "p", "e", "n", "a", "i"]
    test_resumable_iter(ListIterator(OPENAI), OPENAI)

    print("test_list_2 passed!")










  ##############################################
  # PART 3: test multi_file_iterator with jsonl
  ##############################################
  def test_multi_file_1(self):
    iterators = [JsonlFileIterator(f"d{i}.jsonl") for i in range(5)]
    it = MultiFileIterator(iterators)

    assert next(it) == {'item': 1}
    s = it.get_state()
    assert next(it) == {'item': 2}
    assert next(it) == {'item': 3}
    it.set_state(s)
    assert next(it) == {'item': 2}
    assert next(it) == {'item': 3}

    print("test_multi_file_1 passed!")

  def test_multi_file_2(self):
    iterators = [JsonlFileIterator(f"d{i}.jsonl") for i in range(5)]
    it = MultiFileIterator(iterators)

    test_resumable_iter(it, [
        {'item': i} for i in range(1, 9)
    ])

    print("test_multi_file_2 passed!")










  ##############################################
  # PART 4: test efficient iterator multi file
  ##############################################
  def test_multi_file_efficient(self):
    filenames = [f"d{i}.jsonl" for i in range(5)]
    it = EfficientMultiFileIterator(filenames)

    assert next(it) == {'item': 1}
    s = it.get_state()
    assert next(it) == {'item': 2}
    assert next(it) == {'item': 3}
    it.set_state(s)
    assert next(it) == {'item': 2}
    assert next(it) == {'item': 3}

    print("test_multi_file_efficient passed!")

if __name__ == '__main__':
    unittest.main()

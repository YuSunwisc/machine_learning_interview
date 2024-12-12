from b_resumable_iterator import ResumableIterator
from jsonl_file_iterator import JsonlFileIterator

class MultiFileIterator(ResumableIterator):
  def __init__(self, lst):
    self.lst = lst
    self.index = 0
    self.initial_states = [it.get_state() for it in lst]

  def __next__(self):
    while self.index < len(self.lst):
      inner_iter = self.lst[self.index]

      try:
        return next(inner_iter)
      except StopIteration:
        inner_iter.set_state(self.initial_states[self.index])
        self.index += 1
    raise StopIteration

  def get_state(self):
    return {
      'outer_index': self.index,
      'inner_index': self.lst[self.index].get_state()
      }
  
  def set_state(self, s):


    if self.index < len(self.lst):
      inner_iter = self.lst[self.index]
      inner_iter.set_state(self.initial_states[self.index])

    outer_index = s['outer_index']
    inner_index = s['inner_index']

    if not (0 <= outer_index <= len(self.lst)):
        raise IndexError("Outer index is out of range.")

    self.index = outer_index
    self.lst[self.index].set_state(inner_index) 
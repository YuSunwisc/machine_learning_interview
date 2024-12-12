
######################################################################
# PART 2: Implement a resumable iterator and make sure it passes tests
######################################################################

from b_resumable_iterator import ResumableIterator

class ListIterator(ResumableIterator):
  def __init__(self, lst):
    self.lst = lst
    self.index = 0

  def __next__(self):
    if self.index >= len(self.lst):
      raise StopIteration
    
    res = self.lst[self.index]
    self.index += 1
    return res

  def get_state(self):
    return self.index
  
  def set_state(self, s):
    self.index = s

from b_resumable_iterator import ResumableIterator
from jsonl_file_iterator import JsonlFileIterator
from d_multi_file_iterator import MultiFileIterator

# class EfficientMultiFileIterator(ResumableIterator):
#   def __init__(self, outer_iter):
#     self.outer_iter = outer_iter
#     self.outer_state = outer_iter.get_state()
#     self.outer_initial_state = self.outer_iter.set_state()
#     self.inner_iter = None
#     self.inner_initial_state = None


#   def __next__(self):
#     while True:
#       if self.inner_iter is None:
#         try:
#           self.inner_iter = next(self.outer_iter)
#           self.inner_initial_state = self.inner_iter.get_state()
#         except StopIteration:
#           raise StopIteration
        
#         try:
#           return next(self.inner_iter)
#         except:
#           self.inner_iter.set_state(self.inner_initial_state)
#           self.inner_initial_state = None
#           self.inner_iter = None

#   def get_state(self):
#     return {
#       'outer_state': self.outer_state,
#       'inner_state': self.inner_iter.get_state() if self.inner_iter else None
#       }
  
#   def set_state(self, s):
#     if self.inner_iter:
#       self.inner_iter.set_state(self.inner_initial_state)

#     outer_state = s['outer_state']
#     inner_state = s['inner_state']

#     self.outer_state = outer_state
#     self.inner_iter = inner_state

class EfficientMultiFileIterator(ResumableIterator):
    def __init__(self, lst):

      if isinstance(lst[0], str):
        lst = [JsonlFileIterator(filename) for filename in lst]
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
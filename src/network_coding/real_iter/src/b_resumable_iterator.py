###########################################################
# PART 1: Design a resumable iterator and write a unit test
###########################################################

from abc import ABC, abstractmethod

class ResumableIterator(ABC):
  
  def __iter__(self):
    return self

  @abstractmethod
  def __next__(self):
    pass

  @abstractmethod
  def get_state(self):
    pass
  
  @abstractmethod
  def set_state(self, s):
    pass


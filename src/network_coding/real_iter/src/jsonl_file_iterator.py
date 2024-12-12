from c_list_iterator import ListIterator

def JsonlFileIterator(filename):
  return ListIterator({
    "d0.jsonl": [{"item": 1}, {"item": 2}],
    "d1.jsonl": [{"item": 3}, {"item": 4}, {"item": 5}],
    "d2.jsonl": [{"item": 6}, {"item": 7}],
    "d3.jsonl": [],
    "d4.jsonl": [{"item": 8}],
   }[filename])

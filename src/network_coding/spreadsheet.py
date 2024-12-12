# follow up 1 如何优化
# A1: value 只要不是 none 就存起来直接返回，setValue 的时候的 path 不要用 copy，用 dfs

# follow up 2. 针对多线程的 follow up
# A2: 这里可能需要的就是 paging 的思想 (按照某种方式分页)，
# 把 key space 进行 paging 分别上锁从而能够增加并发程度

# follow up 3. 针对 OOM 的 follow up，
# 一种方法是利用 LRU 等一些方法把不太常用的 key 写到磁盘里面去节省空间。
# 简单的方法就是直接 append 到一个 file 里面去，然后需要查找的时候就线性查找; 
# 如果需要更加优化的方法的话，可能要考虑实现简单版本的 SST (SST 文件（Sorted String Table）

# SST 是一种结构化的磁盘存储方案，广泛用于 LSM（Log-Structured Merge）树等数据库系统中。
# 数据以有序的方式存储在文件中，便于快速查找。
# 实现方法：
# 数据按键排序后写入文件。
# 查找时，通过二分查找或索引快速定位键。
# 新数据可以以增量的形式写入新的 SST 文件，避免频繁修改已有文件。)

import unittest
from typing import Dict, Set, Optional

# Implementation
cell_dict: Dict[str, 'Cell'] = {}

class Cell:
    def __init__(self, value=None, child1=None, child2=None):
        self.value = value
        if (child1 and not child2) or (child2 and not child1):
            raise ValueError("Must include both child1 and child2")
        if value and child1:
            raise ValueError("Must input only value of 2 children")
        self.child1 = child1
        self.child2 = child2
        self.parents: Set['Cell'] = set()

    def set_parent(self, parent, path: Optional[Set] = None):
        if path is None:
            path = set()
        if self in path:
            raise ValueError("There is a loop!")
        path.add(self)

        self.parents.add(parent)

        # 递归 dfs 检查所有自上而下的 parent 路径是否是合理的

        # if self.child1 and self.child2:
        #     cell_dict[self.child1].set_parent(self, path.copy())  # 使用路径的副本
        #     cell_dict[self.child2].set_parent(self, path.copy())

        try:
            if self.child1 and self.child2:
                cell_dict[self.child1].set_parent(self, path)
                cell_dict[self.child2].set_parent(self, path)
        finally:
            path.remove(self)  # 回溯时移除当前节点


    def invalidate(self):
        self.value = None
        for parent in self.parents:
            parent.invalidate()

    def get_value(self):
        if self.value is not None:
            return self.value

        v = cell_dict[self.child1].get_value() + cell_dict[self.child2].get_value()
        self.value = v
        return v

class SpreadSheet:
    def set_cell(self, key, cell):
        if key in cell_dict:
            cell_dict[key].invalidate()
        cell_dict[key] = cell
        if cell.child1 and cell.child2:
            cell_dict[cell.child1].set_parent(cell)
            cell_dict[cell.child2].set_parent(cell)

    def get_cell_value(self, key):
        return cell_dict[key].get_value()

# Unit Tests
class TestSpreadSheet(unittest.TestCase):
    def setUp(self):
        global cell_dict
        cell_dict = {}  # Reset the global dictionary for each test case
        self.spreadsheet = SpreadSheet()

    def test_simple_dependency(self):
        self.spreadsheet.set_cell("A", Cell(6))
        self.spreadsheet.set_cell("B", Cell(7))
        self.spreadsheet.set_cell("C", Cell(value=None, child1="A", child2="B"))
        self.assertEqual(self.spreadsheet.get_cell_value("C"), 13)
        self.spreadsheet.set_cell("A", Cell(5))
        self.assertEqual(self.spreadsheet.get_cell_value("C"), 12)

    def test_multiple_dependency(self):
        self.spreadsheet.set_cell("D", Cell(10))
        self.spreadsheet.set_cell("E", Cell(15))
        self.spreadsheet.set_cell("F", Cell(value=None, child1="D", child2="E"))
        self.assertEqual(self.spreadsheet.get_cell_value("F"), 25)
        self.spreadsheet.set_cell("G", Cell(value=None, child1="D", child2="F"))
        self.assertEqual(self.spreadsheet.get_cell_value("G"), 35)
        
        self.spreadsheet.set_cell("D", Cell(20))
        self.assertEqual(self.spreadsheet.get_cell_value("F"), 35)
        self.assertEqual(self.spreadsheet.get_cell_value("G"), 55)


    def test_circular_dependency(self):
        self.spreadsheet.set_cell("D", Cell(10))
        self.spreadsheet.set_cell("E", Cell(15))
        self.spreadsheet.set_cell("F", Cell(value=None, child1="D", child2="E"))
        with self.assertRaises(ValueError) as context:
            self.spreadsheet.set_cell("D", Cell(value=None, child1="F", child2="E"))
        self.assertEqual(str(context.exception), "There is a loop!")

if __name__ == "__main__":
    unittest.main()

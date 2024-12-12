from typing import Dict, Set

cell_dict: Dict[str, 'Cell'] = {}

#         parent
#           |
#         Cell
#       /       \
#  child1       child2

class Cell:
    def __init__(self, value=None, child1=None, child2=None):
        self.value = value
        self.child1 = child1 #只是 str!
        self.child2 = child2 #只是 str!
        self.parents: Set['Cell'] = set()
        
    # set part
    def set_parent(self, parent, visitied=None):
        '''链接所有的 parent'''

        #检查是否有 loop
        if not visited:
            visited = set()
        if parent in visited:
            raise ValueError("There is a loop!")
        visited.add(parent)

        # 主体
        self.parents.add(parent)

        # 从上到下递归检查， self 变成新的 parent，这里只是检查，但是也将原来的 parent 值重新赋值了一遍
        if self.child1 and self.child2:
            cell_dict[self.child1].set_parent(self, visited)
            cell_dict[self.child2].set_parent(self, visited)



    def invalidate(self):
        self.value = None
        for parent in self.parents:
            parent.invalidate()

    # get part
    def get_value(self):
        ''' 返回 cell 的值 '''
        if self.value:
            return self.value
        
        v = cell_dict[self.child1].get_value() + cell_dict[self.child2].get_value()
        self.value = v
        return v
    


class SpreadSheet:
    
    # set part
    def set_cell(self, key, cell):
        # 更新所有 parent
        if key in cell_dict:
            cell_dict[key].invalidate()
        cell_dict[key] = cell

        if cell.child1 and cell.child2:
            cell_dict[child1].set_parent(cell)
            cell_dict[child2].set_parent(cell)

    # get part
    def get_cell_value(self, key):
        return cell_dict[key].get_value()
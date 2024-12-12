# follow up是怎么快速的query，
# A1: 构建 query to data 的一个reverted index
# A2: cache 高频词汇
# A3: 分区查找，感觉不太时候 in memory


from collections import defaultdict
from typing import List, Dict, Any, Callable
from typing import Tuple
import unittest


class InMemoryDB:
    def __init__(self):
        """
        初始化数据库，使用一个 defaultdict 存储表数据。
        每个表是一个字典，键是列名，值是一个列表，表示该列的所有值。
        """
        self.tables = defaultdict(lambda: defaultdict(list))

    def insert(self, table: str, record: Dict[str, Any]) -> None:
        """
        插入一条记录到指定表中。如果记录为空则跳过。
        """
        if not record:
            return
        for col, value in record.items():
            self.tables[table][col].append(value)

    def query(
        self,
        table: str,
        where: Callable[[Dict[str, Any]], bool] = None,
        order_by: List[str] = None,
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        查询指定表中的数据，支持条件查询和排序。
        :param table: 表名
        :param where: 条件函数，接受一条记录返回布尔值
        :param order_by: 排序列的列表，按优先级排序
        :param ascending: 是否升序
        :return: 满足条件的记录列表
        """
        if table not in self.tables:
            raise ValueError(f"Table {table} doesn't exist!")

        if not table:
            return []

        # 提取表中的所有数据
        columns = self.tables[table]
        # num_rows = len(list(columns.values())) # 不能用，因为 columns.values() 是所有的 rows，我们要取其中一个
        num_rows = len(next(iter(columns.values()), []))

        
        records = [
            {col: columns[col][i] for col in columns}
            for i in range(num_rows)
        ]

        # 应用 WHERE 条件
        if where:
            records = [record for record in records if where(record)]

        # 应用 ORDER BY 排序
        if order_by:
            records.sort(
                key=lambda record: tuple(record[col] for col in order_by),
                reverse=not ascending,
            )

        return records

    def join(
        self,
        table1: str,
        table2: str,
        on: Tuple[str, str],
        join_type: str = "inner",
    ) -> List[Dict[str, Any]]:
        """
        在两个表之间进行连接操作。
        :param table1: 左表的名称
        :param table2: 右表的名称
        :param on: 连接条件，元组 (table1_column, table2_column)
        :param join_type: 连接类型，支持 "inner" 和 "left"
        :return: 合并后的记录列表
        """
        if table1 not in self.tables or table2 not in self.tables:
            raise ValueError("One or both tables do not exist!")

        # 提取两张表的数据
        records1 = self.query(table1)
        records2 = self.query(table2)

        col1, col2 = on

        # 存储结果
        result = []

        for r1 in records1:
            matched = False
            for r2 in records2:
                if r1[col1] == r2[col2]:
                    result.append({**r1, **r2})
                    matched = True
            if join_type == "left" and not matched:
                result.append({**r1, **{col: None for col in records2[0].keys()}})

        return result


# 单元测试，包含所有边界条件
class TestInMemoryDB(unittest.TestCase):
    def setUp(self):
        """
        初始化数据库和测试数据
        """
        self.db = InMemoryDB()
        # 插入用户表数据
        self.db.insert("users", {"id": 1, "name": "Alice", "age": 30})
        self.db.insert("users", {"id": 2, "name": "Bob", "age": 25})
        self.db.insert("users", {"id": 3, "name": "Chole", "age": 35})
        self.db.insert("users", {"id": 4, "name": "David", "age": 40})
        self.db.insert("users", {"id": 5, "name": "Eve", "age": 22})
        self.db.insert("empty table", {})

        # 插入订单表数据
        self.db.insert("orders", {"id": 1, "product": "Book", "amount": 100})
        self.db.insert("orders", {"id": 2, "product": "Laptop", "amount": 200})
        self.db.insert("orders", {"id": 4, "product": "Tablet", "amount": 300})

    def test_inner_join(self):
        """测试内连接"""
        result = self.db.join("users", "orders", on=("id", "id"), join_type="inner")
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["name"], "Alice")
        self.assertEqual(result[0]["product"], "Book")

    def test_left_join(self):
        """测试左连接"""
        result = self.db.join("users", "orders", on=("id", "id"), join_type="left")
        self.assertEqual(len(result), 5)
        self.assertEqual(result[2]["name"], "Chole")
        self.assertIsNone(result[2]["product"])


    def test_query_all(self):
        """测试查询所有记录"""
        result = self.db.query("users")
        self.assertEqual(len(result), 5)

    def test_query_with_where(self):
        """测试带 WHERE 条件的查询"""
        result = self.db.query("users", where=lambda r: r["age"] > 30)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r["age"] > 30 for r in result))

    def test_query_with_multiple_where(self):
        """测试带多个条件的查询"""
        result = self.db.query(
            "users", where=lambda r: r["age"] > 20 and r["name"].startswith("A")
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "Alice")

    def test_query_with_order_by(self):
        """测试带 ORDER BY 的查询"""
        result = self.db.query("users", order_by=["age"])
        self.assertEqual(result[0]["age"], 22)
        self.assertEqual(result[-1]["age"], 40)

    def test_query_with_order_by_descending(self):
        """测试带降序 ORDER BY 的查询"""
        result = self.db.query("users", order_by=["age"], ascending=False)
        self.assertEqual(result[0]["age"], 40)
        self.assertEqual(result[-1]["age"], 22)

    def test_query_with_multiple_order_by(self):
        """测试带多个列 ORDER BY 的查询"""
        self.db.insert("users", {"id": 6, "name": "Aaron", "age": 25})
        result = self.db.query("users", order_by=["age", "name"])
        self.assertEqual(result[0]["name"], "Eve")  # 年龄最小
        self.assertEqual(result[1]["name"], "Aaron")  # 同龄时按名字排序

    def test_empty_table(self):
        """测试空表查询"""
        with self.assertRaises(ValueError):
            self.db.query("empty table")


    def test_invalid_table(self):
        """测试查询不存在的表"""
        with self.assertRaises(ValueError):
            self.db.query("nonexistent")

if __name__ == "__main__":
    unittest.main()

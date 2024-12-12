# follow up 如何优化
# 利用 SortedDict 和 irange(min, max) 优化，所有的时间复杂度降到 log(n + k)

import unittest
from bisect import bisect_left, insort

# class GPUCredit:
#     def __init__(self):
#         self.bank = []

#     def addCredit(self, creditID: str, amount: int, timestamp: int, expiration: int) -> None:
#         # A credit is an offering of GPU balance that expires after some expiration-time. The credit can be used only during [timestamp, timestamp + expiration]. **Check with your interviewer whether this period is inclusive '[]' or exclusive '()'. Examples given were inclusive.** A credit can be repeatedly used until expiration.
#         if timestamp > expiration:
#             raise ValueError("Expiration must be larger or equal to timestamp!")
        
#         credit_list = [expiration, timestamp, amount, creditID]
#         pos = bisect_left(self.bank, credit_list)
#         insort(self.bank, credit_list)

#     def getBalance(self, timestamp: int): # -> int | None:
#         # return the balance remaining on the account at the timestamp, return None if there are no credit left. Note, balance cannot be negative. See edge case below.
#         remain = 0
#         pos = bisect_left(self.bank, [timestamp, 0, 0, ''])

#         for i in range(pos, len(self.bank)):
#             remain += self.bank[i][2]

#         return remain if remain != 0 else None
#     def useCredit(self, timestamp: int, amount: int) -> None:
#         pos = bisect_left(self.bank, [timestamp, 0, 0, ''])

#         for i in range(pos, len(self.bank)):
#             if self.bank[i][1] <= timestamp:
#                 if not amount:
#                     break
#                 else:
#                     subtractor = min(self.bank[i][2], amount)
#                     self.bank[i][2] -= subtractor
#                     amount -= subtractor




from sortedcontainers import SortedDict

class GPUCredit:
    def __init__(self):
        self.bank = SortedDict()

    def addCredit(self, creditID: str, amount: int, timestamp: int, expiration: int) -> None:
        if timestamp > expiration:
            raise ValueError("Timestamp cannot be after expiration!")

        key = (expiration, timestamp, creditID)
        self.bank[key] = self.bank.get(key, 0) + amount

    def getBalance(self, timestamp: int):
        remain = 0
        for (exp, start, creditID) in self.bank.irange((timestamp, 0, ''), (float('inf'), float('inf'), '')):
            if start <= timestamp <= exp:
                remain += self.bank[(exp, start, creditID)]
        return remain if remain > 0 else None

    def useCredit(self, timestamp: int, amount: int) -> None:
        keys_to_remove = []
        for (exp, start, creditID) in self.bank.irange((timestamp, 0, ''), (float('inf'), float('inf'), '')):
            if start <= timestamp <= exp and amount > 0:
                balance = self.bank[(exp, start, creditID)]
                used = min(balance, amount)
                self.bank[(exp, start, creditID)] -= used
                amount -= used
                if self.bank[(exp, start, creditID)] == 0:
                    keys_to_remove.append((exp, start, creditID))

        for key in keys_to_remove:
            del self.bank[key]

class TestGPUCredit(unittest.TestCase):
    def test_valid_input(self):
        gpu = GPUCredit()

        with self.assertRaises(ValueError):
            gpu.addCredit('OpenAI', 10, 20241203, 20241202)
        
    def test_valid_getBalance(self):
        gpu = GPUCredit()

        self.assertIsNone(gpu.getBalance(20241202))

        gpu.addCredit('OpenAI', 10, 20241203, 20241206)


        self.assertEqual(gpu.getBalance(20241203), 10)
        self.assertEqual(gpu.getBalance(20241204), 10)
        self.assertEqual(gpu.getBalance(20241205), 10)
        self.assertEqual(gpu.getBalance(20241206), 10)
        self.assertIsNone(gpu.getBalance(20241207))


    def test_valid_useCredit(self):
        gpu = GPUCredit()

        gpu.addCredit('OpenAI', 10, 20241203, 20241206)

        gpu.useCredit(20241204, 6)

        self.assertEqual(gpu.getBalance(20241204), 4)
        self.assertEqual(gpu.getBalance(20241205), 4)
        self.assertEqual(gpu.getBalance(20241206), 4)
        self.assertIsNone(gpu.getBalance(20241207))

    def test_valid_overusage_1(self):
        gpu = GPUCredit()

        gpu.addCredit('OpenAI', 10, 20241203, 20241206)

        gpu.useCredit(20241204, 10000)

   
        self.assertIsNone(gpu.getBalance(20241204))
        self.assertIsNone(gpu.getBalance(20241205))
        self.assertIsNone(gpu.getBalance(20241206))
        self.assertIsNone(gpu.getBalance(20241207))

    def test_valid_overusage_2(self):
        gpu = GPUCredit()

        gpu.addCredit('OpenAI', 10, 20241203, 20241206)

        gpu.useCredit(20241204, 10000)

        self.assertIsNone(gpu.getBalance(20241204))

        gpu.addCredit('Microsoft', 10, 20241204, 20241209)

        self.assertEqual(gpu.getBalance(20241204), 10)

        gpu.useCredit(20241204, 6)

        self.assertEqual(gpu.getBalance(20241204), 4)
        self.assertEqual(gpu.getBalance(20241209), 4)

        self.assertIsNone(gpu.getBalance(20241210))

    def test_valid_greedy(self):
        gpu = GPUCredit()

        gpu.addCredit('OpenAI', 30, 20241203, 20241206)

        gpu.addCredit('Microsoft', 10, 20241204, 20241209)

        gpu.useCredit(20241204, 30)
        gpu.useCredit(20241208, 9)

        self.assertEqual(gpu.getBalance(20241208), 1)

if __name__ == '__main__':
    unittest.main()
        
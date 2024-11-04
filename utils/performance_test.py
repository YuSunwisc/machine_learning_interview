# performance_test.py

import timeit
from memory_profiler import memory_usage

class PerformanceTest:
    @staticmethod
    def estimate_time_complexity(func, *args, repeat=10):
        """
        Estimate time complexity by measuring the execution time over several runs.
        :param func: function to test
        :param args: arguments to pass to the function
        :param repeat: number of repetitions for averaging
        :return: average execution time
        """
        try:
            timer = timeit.Timer(lambda: func(*args))
            time_taken = timer.timeit(number=repeat) / repeat
            print(f"Average execution time over {repeat} runs: {time_taken:.6f} seconds")
            return time_taken
        except Exception as e:
            print(f"Error in time complexity estimation: {e}")
            return None

    @staticmethod
    def estimate_space_complexity(func, *args):
        """
        Estimate space complexity by measuring memory usage.
        :param func: function to test
        :param args: arguments to pass to the function
        :return: peak memory usage during function execution
        """
        try:
            mem_usage = memory_usage((func, args), interval=0.1)
            peak_memory = max(mem_usage) - min(mem_usage)
            print(f"Peak memory usage: {peak_memory:.6f} MiB")
            return peak_memory
        except Exception as e:
            print(f"Error in space complexity estimation: {e}")
            return None

# Simple test when running this file directly
if __name__ == "__main__":
    # Example function for testing
    def sample_function(n):
        return [i ** 2 for i in range(n)]

    # Test parameters
    test_input = 1000

    print("Running performance tests on sample_function...")

    # Estimate time complexity
    PerformanceTest.estimate_time_complexity(sample_function, test_input)

    # Estimate space complexity
    PerformanceTest.estimate_space_complexity(sample_function, test_input)

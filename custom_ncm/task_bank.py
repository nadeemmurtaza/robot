"""
Problem and Unit-Test Bank for Error-Free Code Self-Training.
Provides executable tasks with ground-truth test cases to automatically verify generated code.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict

@dataclass
class CodingTask:
    task_id: str
    prompt: str
    func_name: str
    reference_body: str
    test_cases: List[Tuple[Any, Any]] # [(args_tuple_or_single_arg, expected_return), ...]


TASK_BANK: List[CodingTask] = [
    CodingTask(
        task_id="sum_list",
        prompt="# Compute sum of all elements in list\ndef sum_list(nums):\n",
        func_name="sum_list",
        reference_body="    total = 0\n    for x in nums:\n        total += x\n    return total\n",
        test_cases=[
            ([1, 2, 3, 4], 10),
            ([], 0),
            ([-5, 5, 10], 10),
            ([100], 100)
        ]
    ),
    CodingTask(
        task_id="get_max",
        prompt="# Return the maximum value in a list\ndef get_max(nums):\n",
        func_name="get_max",
        reference_body="    if not nums:\n        return None\n    m = nums[0]\n    for x in nums[1:]:\n        if x > m:\n            m = x\n    return m\n",
        test_cases=[
            ([1, 5, 3, 9, 2], 9),
            ([-10, -5, -20], -5),
            ([42], 42),
            ([], None)
        ]
    ),
    CodingTask(
        task_id="get_min",
        prompt="# Return the minimum value in a list\ndef get_min(nums):\n",
        func_name="get_min",
        reference_body="    if not nums:\n        return None\n    m = nums[0]\n    for x in nums[1:]:\n        if x < m:\n            m = x\n    return m\n",
        test_cases=[
            ([1, 5, 3, 9, 2], 1),
            ([-10, -5, -20], -20),
            ([42], 42),
            ([], None)
        ]
    ),
    CodingTask(
        task_id="factorial",
        prompt="# Compute factorial of an integer n\ndef factorial(n):\n",
        func_name="factorial",
        reference_body="    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n",
        test_cases=[
            (0, 1),
            (1, 1),
            (5, 120),
            (6, 720)
        ]
    ),
    CodingTask(
        task_id="is_prime",
        prompt="# Check if an integer is prime\ndef is_prime(n):\n",
        func_name="is_prime",
        reference_body="    if n < 2:\n        return False\n    for d in range(2, int(n**0.5) + 1):\n        if n % d == 0:\n            return False\n    return True\n",
        test_cases=[
            (2, True),
            (3, True),
            (4, False),
            (17, True),
            (1, False),
            (0, False)
        ]
    ),
    CodingTask(
        task_id="fibonacci",
        prompt="# Generate first n Fibonacci numbers\ndef fibonacci(n):\n",
        func_name="fibonacci",
        reference_body="    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]\n",
        test_cases=[
            (1, [0]),
            (2, [0, 1]),
            (5, [0, 1, 1, 2, 3]),
            (0, [])
        ]
    ),
    CodingTask(
        task_id="gcd",
        prompt="# Compute greatest common divisor\ndef gcd(a, b):\n",
        func_name="gcd",
        reference_body="    while b:\n        a, b = b, a % b\n    return a\n",
        test_cases=[
            ((12, 18), 6),
            ((7, 13), 1),
            ((20, 5), 5),
            ((100, 25), 25)
        ]
    ),
    CodingTask(
        task_id="is_palindrome",
        prompt="# Check if a string is a palindrome\ndef is_palindrome(s):\n",
        func_name="is_palindrome",
        reference_body="    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n",
        test_cases=[
            ("racecar", True),
            ("hello", False),
            ("A man a plan a canal Panama", True),
            ("", True)
        ]
    ),
    CodingTask(
        task_id="filter_evens",
        prompt="# Filter all even numbers from list\ndef filter_evens(nums):\n",
        func_name="filter_evens",
        reference_body="    return [x for x in nums if x % 2 == 0]\n",
        test_cases=[
            ([1, 2, 3, 4, 5, 6], [2, 4, 6]),
            ([1, 3, 5], []),
            ([], []),
            ([2, 4], [2, 4])
        ]
    ),
    CodingTask(
        task_id="reverse_words",
        prompt="# Reverse the order of words in text\ndef reverse_words(text):\n",
        func_name="reverse_words",
        reference_body="    words = text.split()\n    return ' '.join(reversed(words))\n",
        test_cases=[
            ("hello world", "world hello"),
            ("the quick brown fox", "fox brown quick the"),
            ("code", "code"),
            ("", "")
        ]
    ),
    CodingTask(
        task_id="count_vowels",
        prompt="# Count vowels in a given text\ndef count_vowels(text):\n",
        func_name="count_vowels",
        reference_body="    vowels = set('aeiouAEIOU')\n    return sum(1 for ch in text if ch in vowels)\n",
        test_cases=[
            ("apple", 2),
            ("rhythm", 0),
            ("Antigravity", 5),
            ("", 0)
        ]
    ),
    CodingTask(
        task_id="average",
        prompt="# Calculate average of numbers in list\ndef average(nums):\n",
        func_name="average",
        reference_body="    if not nums:\n        return 0.0\n    return sum(nums) / len(nums)\n",
        test_cases=[
            ([1, 2, 3, 4, 5], 3.0),
            ([10, 20], 15.0),
            ([], 0.0)
        ]
    ),
    CodingTask(
        task_id="remove_duplicates",
        prompt="# Remove duplicates while preserving order\ndef remove_duplicates(lst):\n",
        func_name="remove_duplicates",
        reference_body="    seen = set()\n    out = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
        test_cases=[
            ([1, 2, 2, 3, 1, 4], [1, 2, 3, 4]),
            ([5, 5, 5], [5]),
            ([], [])
        ]
    ),
    CodingTask(
        task_id="binary_search",
        prompt="# Perform binary search on sorted list\ndef binary_search(arr, target):\n",
        func_name="binary_search",
        reference_body="    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        test_cases=[
            (([1, 3, 5, 7, 9], 5), 2),
            (([1, 3, 5, 7, 9], 1), 0),
            (([1, 3, 5, 7, 9], 9), 4),
            (([1, 3, 5, 7, 9], 2), -1),
            (([], 4), -1)
        ]
    )
]

def get_all_tasks() -> List[CodingTask]:
    return list(TASK_BANK)

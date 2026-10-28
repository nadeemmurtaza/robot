"""
Comprehensive Python Master Curriculum for Teaching the Next-Concept Model.
10 Progressive Chapters:
- Chapter 1: Basic Arithmetic & Math
- Chapter 2: Conditions & Logic
- Chapter 3: Strings & Text Operations
- Chapter 4: Lists & Iterations
- Chapter 5: Common Algorithms & Problem Solving
- Chapter 6: Dictionaries & Key-Value Maps
- Chapter 7: Sets & Unique Collections
- Chapter 8: Recursion & Mathematical Sequences
- Chapter 9: List & Data Comprehensions
- Chapter 10: Sorting & Two-Pointer Patterns
"""
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

@dataclass
class Lesson:
    chapter: str
    topic: str
    prompt: str
    func_name: str
    target_code: str
    test_cases: List[Tuple[Any, Any]]


PYTHON_CURRICULUM: List[Lesson] = [
    # =========================================================================
    # CHAPTER 1: BASIC ARITHMETIC & OPERATORS
    # =========================================================================
    Lesson(
        chapter="Chapter 1: Arithmetic & Math",
        topic="Addition",
        prompt="# Return sum of two numbers\ndef add(a, b):\n",
        func_name="add",
        target_code="    return a + b\n",
        test_cases=[((2, 3), 5), ((-1, 1), 0), ((0, 0), 0)]
    ),
    Lesson(
        chapter="Chapter 1: Arithmetic & Math",
        topic="Multiplication",
        prompt="# Return product of two numbers\ndef multiply(a, b):\n",
        func_name="multiply",
        target_code="    return a * b\n",
        test_cases=[((4, 5), 20), ((-2, 3), -6), ((0, 10), 0)]
    ),
    Lesson(
        chapter="Chapter 1: Arithmetic & Math",
        topic="Squaring",
        prompt="# Return the square of a number\ndef square(x):\n",
        func_name="square",
        target_code="    return x * x\n",
        test_cases=[((3,), 9), ((5,), 25), ((-4,), 16), ((0,), 0)]
    ),
    Lesson(
        chapter="Chapter 1: Arithmetic & Math",
        topic="Integer Division and Modulo",
        prompt="# Return remainder of a divided by b\ndef remainder(a, b):\n",
        func_name="remainder",
        target_code="    return a % b\n",
        test_cases=[((10, 3), 1), ((20, 5), 0), ((7, 2), 1)]
    ),

    # =========================================================================
    # CHAPTER 2: CONDITIONS & LOGIC
    # =========================================================================
    Lesson(
        chapter="Chapter 2: Conditions & Logic",
        topic="Even Number Check",
        prompt="# Check if a number is even\ndef is_even(n):\n",
        func_name="is_even",
        target_code="    return n % 2 == 0\n",
        test_cases=[((4,), True), ((7,), False), ((0,), True), ((-2,), True)]
    ),
    Lesson(
        chapter="Chapter 2: Conditions & Logic",
        topic="Positive Number Check",
        prompt="# Check if number is strictly positive\ndef is_positive(n):\n",
        func_name="is_positive",
        target_code="    return n > 0\n",
        test_cases=[((5,), True), ((-3,), False), ((0,), False)]
    ),
    Lesson(
        chapter="Chapter 2: Conditions & Logic",
        topic="Maximum of Two Values",
        prompt="# Return larger of two numbers\ndef max_two(a, b):\n",
        func_name="max_two",
        target_code="    return a if a > b else b\n",
        test_cases=[((10, 20), 20), ((-5, -10), -5), ((7, 7), 7)]
    ),
    Lesson(
        chapter="Chapter 2: Conditions & Logic",
        topic="Absolute Value",
        prompt="# Return absolute value of a number\ndef absolute_val(x):\n",
        func_name="absolute_val",
        target_code="    return x if x >= 0 else -x\n",
        test_cases=[((5,), 5), ((-12,), 12), ((0,), 0)]
    ),

    # =========================================================================
    # CHAPTER 3: STRINGS & TEXT OPERATIONS
    # =========================================================================
    Lesson(
        chapter="Chapter 3: Strings & Text",
        topic="String Length",
        prompt="# Return the length of a string\ndef string_length(s):\n",
        func_name="string_length",
        target_code="    return len(s)\n",
        test_cases=[(("hello",), 5), (("",), 0), (("Python",), 6)]
    ),
    Lesson(
        chapter="Chapter 3: Strings & Text",
        topic="Greeting Concatenation",
        prompt="# Return greeting for given name\ndef greet(name):\n",
        func_name="greet",
        target_code="    return 'Hello, ' + name + '!'\n",
        test_cases=[(("Alice",), "Hello, Alice!"), (("World",), "Hello, World!")]
    ),
    Lesson(
        chapter="Chapter 3: Strings & Text",
        topic="String Repetition",
        prompt="# Repeat string s n times\ndef repeat(s, n):\n",
        func_name="repeat",
        target_code="    return s * n\n",
        test_cases=[(("hi", 3), "hihihi"), (("a", 0), "")]
    ),
    Lesson(
        chapter="Chapter 3: Strings & Text",
        topic="Uppercase Conversion",
        prompt="# Convert string to all uppercase\ndef to_upper(s):\n",
        func_name="to_upper",
        target_code="    return s.upper()\n",
        test_cases=[(("code",), "CODE"), (("Hello World",), "HELLO WORLD")]
    ),

    # =========================================================================
    # CHAPTER 4: LISTS & ITERATION
    # =========================================================================
    Lesson(
        chapter="Chapter 4: Lists & Iterations",
        topic="List Sum",
        prompt="# Return sum of list of numbers\ndef sum_numbers(nums):\n",
        func_name="sum_numbers",
        target_code="    total = 0\n    for x in nums:\n        total += x\n    return total\n",
        test_cases=[(([1, 2, 3],), 6), (([],), 0), (([-5, 5],), 0)]
    ),
    Lesson(
        chapter="Chapter 4: Lists & Iterations",
        topic="First Element Safe Lookup",
        prompt="# Return first element or None if empty\ndef first_item(lst):\n",
        func_name="first_item",
        target_code="    return lst[0] if lst else None\n",
        test_cases=[(([10, 20],), 10), (([],), None), ((["a"],), "a")]
    ),
    Lesson(
        chapter="Chapter 4: Lists & Iterations",
        topic="Last Element Safe Lookup",
        prompt="# Return last element or None if empty\ndef last_item(lst):\n",
        func_name="last_item",
        target_code="    return lst[-1] if lst else None\n",
        test_cases=[(([10, 20],), 20), (([],), None), ((["z"],), "z")]
    ),
    Lesson(
        chapter="Chapter 4: Lists & Iterations",
        topic="Double Elements in List",
        prompt="# Return new list with doubled elements\ndef double_all(nums):\n",
        func_name="double_all",
        target_code="    return [x * 2 for x in nums]\n",
        test_cases=[(([1, 2, 3],), [2, 4, 6]), (([],), [])]
    ),

    # =========================================================================
    # CHAPTER 5: ALGORITHMS & LOGIC
    # =========================================================================
    Lesson(
        chapter="Chapter 5: Algorithms & Logic",
        topic="Factorial Calculation",
        prompt="# Compute factorial of integer n\ndef factorial(n):\n",
        func_name="factorial",
        target_code="    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n",
        test_cases=[((0,), 1), ((1,), 1), ((5,), 120), ((6,), 720)]
    ),
    Lesson(
        chapter="Chapter 5: Algorithms & Logic",
        topic="Palindrome Check",
        prompt="# Check if string is palindrome\ndef is_palindrome(s):\n",
        func_name="is_palindrome",
        target_code="    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]\n",
        test_cases=[(("radar",), True), (("hello",), False), (("A man a plan a canal Panama",), True)]
    ),
    Lesson(
        chapter="Chapter 5: Algorithms & Logic",
        topic="Find Element Linear Search",
        prompt="# Return index of target in list or -1\ndef find_index(lst, target):\n",
        func_name="find_index",
        target_code="    for i, x in enumerate(lst):\n        if x == target:\n            return i\n    return -1\n",
        test_cases=[(([10, 20, 30], 20), 1), (([10, 20], 99), -1), (([], 5), -1)]
    ),

    # =========================================================================
    # CHAPTER 6: DICTIONARIES & KEY-VALUE MAPS
    # =========================================================================
    Lesson(
        chapter="Chapter 6: Dictionaries & Maps",
        topic="Dictionary Safe Get",
        prompt="# Get key from dict or return default\ndef safe_get(d, key, default):\n",
        func_name="safe_get",
        target_code="    return d.get(key, default)\n",
        test_cases=[(({"a": 1}, "a", 0), 1), (({"a": 1}, "b", 99), 99), (({}, "x", None), None)]
    ),
    Lesson(
        chapter="Chapter 6: Dictionaries & Maps",
        topic="Invert Dictionary",
        prompt="# Invert dictionary mapping values to keys\ndef invert_map(d):\n",
        func_name="invert_map",
        target_code="    return {v: k for k, v in d.items()}\n",
        test_cases=[(({"apple": "fruit"},), {"fruit": "apple"}), (({},), {})]
    ),
    Lesson(
        chapter="Chapter 6: Dictionaries & Maps",
        topic="Frequency Counter",
        prompt="# Count frequency of items in list\ndef count_frequencies(items):\n",
        func_name="count_frequencies",
        target_code="    counts = {}\n    for x in items:\n        counts[x] = counts.get(x, 0) + 1\n    return counts\n",
        test_cases=[((["a", "b", "a"],), {"a": 2, "b": 1}), (([],), {})]
    ),
    Lesson(
        chapter="Chapter 6: Dictionaries & Maps",
        topic="Key with Maximum Value",
        prompt="# Return key with highest numeric value\ndef max_key(d):\n",
        func_name="max_key",
        target_code="    return max(d, key=d.get) if d else None\n",
        test_cases=[(({"a": 10, "b": 50, "c": 30},), "b"), (({},), None)]
    ),

    # =========================================================================
    # CHAPTER 7: SETS & UNIQUE COLLECTIONS
    # =========================================================================
    Lesson(
        chapter="Chapter 7: Sets & Collections",
        topic="Count Unique Elements",
        prompt="# Return count of unique items in list\ndef count_unique(items):\n",
        func_name="count_unique",
        target_code="    return len(set(items))\n",
        test_cases=[(([1, 2, 2, 3, 1],), 3), (([],), 0), ((["a", "a"],), 1)]
    ),
    Lesson(
        chapter="Chapter 7: Sets & Collections",
        topic="Set Intersection",
        prompt="# Return common elements between two lists\ndef common_items(a, b):\n",
        func_name="common_items",
        target_code="    return sorted(list(set(a) & set(b)))\n",
        test_cases=[(([1, 2, 3], [2, 3, 4]), [2, 3]), (([1, 2], [3, 4]), [])]
    ),
    Lesson(
        chapter="Chapter 7: Sets & Collections",
        topic="Subset Check",
        prompt="# Check if list a is subset of list b\ndef is_subset(a, b):\n",
        func_name="is_subset",
        target_code="    return set(a).issubset(set(b))\n",
        test_cases=[(([1, 2], [1, 2, 3]), True), (([1, 4], [1, 2, 3]), False)]
    ),

    # =========================================================================
    # CHAPTER 8: RECURSION & SEQUENCES
    # =========================================================================
    Lesson(
        chapter="Chapter 8: Recursion & Sequences",
        topic="Recursive Sum of Natural Numbers",
        prompt="# Compute sum from 1 to n recursively\ndef sum_to_n(n):\n",
        func_name="sum_to_n",
        target_code="    return 0 if n <= 0 else n + sum_to_n(n - 1)\n",
        test_cases=[((5,), 15), ((1,), 1), ((0,), 0)]
    ),
    Lesson(
        chapter="Chapter 8: Recursion & Sequences",
        topic="Sum of Digits",
        prompt="# Return sum of digits of positive integer\ndef sum_digits(n):\n",
        func_name="sum_digits",
        target_code="    return sum(int(d) for d in str(n))\n",
        test_cases=[((123,), 6), ((405,), 9), ((0,), 0)]
    ),
    Lesson(
        chapter="Chapter 8: Recursion & Sequences",
        topic="Recursive Power",
        prompt="# Compute base raised to exp recursively\ndef power_rec(base, exp):\n",
        func_name="power_rec",
        target_code="    return 1 if exp == 0 else base * power_rec(base, exp - 1)\n",
        test_cases=[((2, 3), 8), ((5, 0), 1), ((3, 2), 9)]
    ),

    # =========================================================================
    # CHAPTER 9: COMPREHENSIONS & TRANSFORMATIONS
    # =========================================================================
    Lesson(
        chapter="Chapter 9: Comprehensions",
        topic="Filter Positive Numbers",
        prompt="# Filter and return only positive numbers\ndef filter_positives(nums):\n",
        func_name="filter_positives",
        target_code="    return [x for x in nums if x > 0]\n",
        test_cases=[(([1, -2, 3, 0, -5],), [1, 3]), (([],), [])]
    ),
    Lesson(
        chapter="Chapter 9: Comprehensions",
        topic="String Word Lengths",
        prompt="# Return list of word lengths for text\ndef word_lengths(text):\n",
        func_name="word_lengths",
        target_code="    return [len(w) for w in text.split()]\n",
        test_cases=[(("hi world code",), [2, 5, 4]), (("",), [])]
    ),
    Lesson(
        chapter="Chapter 9: Comprehensions",
        topic="Flatten 2D Grid",
        prompt="# Flatten 2D grid into single 1D list\ndef flatten_grid(grid):\n",
        func_name="flatten_grid",
        target_code="    return [x for row in grid for x in row]\n",
        test_cases=[(([[1, 2], [3, 4]],), [1, 2, 3, 4]), (([],), [])]
    ),

    # =========================================================================
    # CHAPTER 10: PATTERNS & UTILITIES
    # =========================================================================
    Lesson(
        chapter="Chapter 10: Patterns & Utilities",
        topic="Clamp Value Within Range",
        prompt="# Clamp val to range between low and high\ndef clamp(val, low, high):\n",
        func_name="clamp",
        target_code="    return max(low, min(val, high))\n",
        test_cases=[((5, 0, 10), 5), ((-5, 0, 10), 0), ((15, 0, 10), 10)]
    ),
    Lesson(
        chapter="Chapter 10: Patterns & Utilities",
        topic="Check Anagrams",
        prompt="# Check if two strings are anagrams\ndef are_anagrams(s1, s2):\n",
        func_name="are_anagrams",
        target_code="    return sorted(s1) == sorted(s2)\n",
        test_cases=[(("silent", "listen"), True), (("cat", "rat"), False)]
    ),
    Lesson(
        chapter="Chapter 10: Patterns & Utilities",
        topic="Check Monotonic List",
        prompt="# Check if list is strictly increasing\ndef is_increasing(nums):\n",
        func_name="is_increasing",
        target_code="    return all(nums[i] < nums[i+1] for i in range(len(nums)-1))\n",
        test_cases=[(([1, 3, 5, 7],), True), (([1, 2, 2, 4],), False), (([],), True)]
    )
]

def get_curriculum() -> List[Lesson]:
    return list(PYTHON_CURRICULUM)

def get_chapters() -> Dict[str, List[Lesson]]:
    chapters = {}
    for lesson in PYTHON_CURRICULUM:
        chapters.setdefault(lesson.chapter, []).append(lesson)
    return chapters

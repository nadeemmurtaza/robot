"""
Dataset generator and PyTorch DataLoader for Code Next-Concept Modeling.
Provides structured prompt-completion pairs of Python functions.
"""
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader

# Comprehensive library of diverse, high-quality Python algorithms & utilities
CODE_CORPUS: List[Tuple[str, str]] = [
    # 1. Basic Math & Number Theory
    (
        "# Compute sum of all elements in list\ndef sum_list(nums):\n",
        "    total = 0\n    for x in nums:\n        total += x\n    return total\n"
    ),
    (
        "# Return the maximum value in a list\ndef get_max(nums):\n",
        "    if not nums:\n        return None\n    m = nums[0]\n    for x in nums[1:]:\n        if x > m:\n            m = x\n    return m\n"
    ),
    (
        "# Return the minimum value in a list\ndef get_min(nums):\n",
        "    if not nums:\n        return None\n    m = nums[0]\n    for x in nums[1:]:\n        if x < m:\n            m = x\n    return m\n"
    ),
    (
        "# Compute factorial of an integer n\ndef factorial(n):\n",
        "    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n"
    ),
    (
        "# Check if an integer is prime\ndef is_prime(n):\n",
        "    if n < 2:\n        return False\n    for d in range(2, int(n**0.5) + 1):\n        if n % d == 0:\n            return False\n    return True\n"
    ),
    (
        "# Generate first n Fibonacci numbers\ndef fibonacci(n):\n",
        "    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]\n"
    ),
    (
        "# Compute greatest common divisor\ndef gcd(a, b):\n",
        "    while b:\n        a, b = b, a % b\n    return a\n"
    ),
    (
        "# Compute least common multiple\ndef lcm(a, b):\n",
        "    return abs(a * b) // gcd(a, b)\n"
    ),
    (
        "# Check if integer is a power of two\ndef is_power_of_two(n):\n",
        "    return n > 0 and (n & (n - 1)) == 0\n"
    ),
    (
        "# Compute integer power x raised to y\ndef power(base, exp):\n",
        "    res = 1\n    while exp > 0:\n        if exp % 2 == 1:\n            res *= base\n        base *= base\n        exp //= 2\n    return res\n"
    ),
    (
        "# Calculate average of numbers in list\ndef average(nums):\n",
        "    if not nums:\n        return 0.0\n    return sum(nums) / len(nums)\n"
    ),

    # 2. String Manipulation
    (
        "# Check if a string is a palindrome\ndef is_palindrome(s):\n",
        "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n"
    ),
    (
        "# Reverse the order of words in text\ndef reverse_words(text):\n",
        "    words = text.split()\n    return ' '.join(reversed(words))\n"
    ),
    (
        "# Count vowels in a given text\ndef count_vowels(text):\n",
        "    vowels = set('aeiouAEIOU')\n    return sum(1 for ch in text if ch in vowels)\n"
    ),
    (
        "# Check if two strings are anagrams\ndef are_anagrams(s1, s2):\n",
        "    return sorted(s1) == sorted(s2)\n"
    ),
    (
        "# Count occurrences of character in text\ndef count_char(s, ch):\n",
        "    count = 0\n    for c in s:\n        if c == ch:\n            count += 1\n    return count\n"
    ),
    (
        "# Truncate string to max length with ellipsis\ndef truncate(s, n):\n",
        "    if len(s) <= n:\n        return s\n    return s[:n] + '...'\n"
    ),
    (
        "# Remove punctuation from string\ndef strip_punctuation(s):\n",
        "    return ''.join(c for c in s if c.isalnum() or c.isspace())\n"
    ),

    # 3. List & Array Operations
    (
        "# Filter all even numbers from list\ndef filter_evens(nums):\n",
        "    return [x for x in nums if x % 2 == 0]\n"
    ),
    (
        "# Filter all odd numbers from list\ndef filter_odds(nums):\n",
        "    return [x for x in nums if x % 2 != 0]\n"
    ),
    (
        "# Flatten a 2D matrix into a 1D list\ndef flatten_matrix(matrix):\n",
        "    return [val for row in matrix for val in row]\n"
    ),
    (
        "# Remove duplicates while preserving order\ndef remove_duplicates(lst):\n",
        "    seen = set()\n    out = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n"
    ),
    (
        "# Count frequency of elements in list\ndef count_frequency(items):\n",
        "    counts = {}\n    for x in items:\n        counts[x] = counts.get(x, 0) + 1\n    return counts\n"
    ),
    (
        "# Merge two sorted lists into one\ndef merge_sorted(a, b):\n",
        "    res, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            res.append(a[i])\n            i += 1\n        else:\n            res.append(b[j])\n            j += 1\n    return res + a[i:] + b[j:]\n"
    ),
    (
        "# Chunk a list into batches of size k\ndef chunk_list(lst, k):\n",
        "    return [lst[i:i + k] for i in range(0, len(lst), k)]\n"
    ),
    (
        "# Find intersection of two lists\ndef list_intersection(a, b):\n",
        "    set_b = set(b)\n    return [x for x in a if x in set_b]\n"
    ),

    # 4. Search & Algorithmic Patterns
    (
        "# Perform binary search on sorted list\ndef binary_search(arr, target):\n",
        "    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n"
    ),
    (
        "# Linear search for target index\ndef linear_search(arr, target):\n",
        "    for i, val in enumerate(arr):\n        if val == target:\n            return i\n    return -1\n"
    ),
    (
        "# Check if array is monotonically increasing\ndef is_sorted(arr):\n",
        "    for i in range(len(arr) - 1):\n        if arr[i] > arr[i + 1]:\n            return False\n    return True\n"
    ),
    (
        "# Find index of minimum element in list\ndef argmin(arr):\n",
        "    if not arr:\n        return -1\n    min_idx = 0\n    for i in range(1, len(arr)):\n        if arr[i] < arr[min_idx]:\n            min_idx = i\n    return min_idx\n"
    ),
    (
        "# Find index of maximum element in list\ndef argmax(arr):\n",
        "    if not arr:\n        return -1\n    max_idx = 0\n    for i in range(1, len(arr)):\n        if arr[i] > arr[max_idx]:\n            max_idx = i\n    return max_idx\n"
    ),

    # 5. Dictionaries and Data Structures
    (
        "# Invert dictionary mapping values to keys\ndef invert_dict(d):\n",
        "    return {v: k for k, v in d.items()}\n"
    ),
    (
        "# Find key with highest value in dictionary\ndef key_with_max_val(d):\n",
        "    if not d:\n        return None\n    return max(d, key=d.get)\n"
    ),
    (
        "# Group elements by a key function\ndef group_by(items, key_fn):\n",
        "    groups = {}\n    for x in items:\n        k = key_fn(x)\n        groups.setdefault(k, []).append(x)\n    return groups\n"
    )
]

def generate_augmented_corpus(multiplier: int = 15) -> List[Tuple[str, str]]:
    """
    Augments the base corpus with variable renamings and slight docstring variations.
    """
    augmented = list(CODE_CORPUS)
    var_map = [
        ("nums", "items"), ("nums", "arr"), ("res", "ans"),
        ("total", "acc"), ("m", "best"), ("cleaned", "norm"),
        ("words", "tokens"), ("counts", "freq_map"), ("text", "data")
    ]
    
    for _ in range(multiplier):
        for prefix, body in CODE_CORPUS:
            aug_prefix, aug_body = prefix, body
            if random.random() > 0.4:
                v_src, v_dst = random.choice(var_map)
                aug_prefix = aug_prefix.replace(v_src, v_dst)
                aug_body = aug_body.replace(v_src, v_dst)
            augmented.append((aug_prefix, aug_body))
            
    return augmented


class CodeDataset(Dataset):
    """
    PyTorch Dataset providing:
    - prefix: prompt tokens
    - target: completion tokens
    """
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, prefix_len: int = 32, context_len: int = 64):
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.context_len = context_len
        self.samples = []
        
        pad_id = tokenizer.pad_token_id
        
        for p_str, b_str in pairs:
            p_ids = tokenizer.encode(p_str)
            b_ids = tokenizer.encode(b_str)
            
            # Truncate or pad prefix to fixed prefix_len
            if len(p_ids) > prefix_len:
                p_ids = p_ids[-prefix_len:]
            else:
                p_ids = p_ids + [pad_id] * (prefix_len - len(p_ids))
                
            # Truncate or pad target to fixed context_len
            if len(b_ids) > context_len:
                b_ids = b_ids[:context_len]
            else:
                b_ids = b_ids + [pad_id] * (context_len - len(b_ids))
                
            self.samples.append({
                "prefix": torch.tensor(p_ids, dtype=torch.long),
                "target": torch.tensor(b_ids, dtype=torch.long),
                "prefix_str": p_str,
                "target_str": b_str,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def get_dataloaders(tokenizer, prefix_len: int = 32, context_len: int = 64, batch_size: int = 16):
    data = generate_augmented_corpus(multiplier=15)
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(0.85 * len(data))
    train_pairs = data[:split_idx]
    val_pairs = data[split_idx:]
    
    train_ds = CodeDataset(train_pairs, tokenizer, prefix_len, context_len)
    val_ds = CodeDataset(val_pairs, tokenizer, prefix_len, context_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

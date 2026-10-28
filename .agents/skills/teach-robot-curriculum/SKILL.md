---
name: teach-robot-curriculum
description: >-
  Interactive and automated teaching workflows for Code Robot. Use when the user asks to teach,
  mentor, or guide Code Robot through the 10-chapter Python curriculum, run the developmental nursery,
  or administer interactive coding lessons with real-time AST feedback.
---

# 🎓 Teach Code Robot Curriculum Skill

This skill provides step-by-step procedures for teaching and mentoring **Code Robot** using developmental curriculum learning, progressive difficulty scaling, and compiler-verified feedback loops.

---

## 🧭 Teaching Progression Overview

Code Robot learns Python like a human child through structured developmental stages:

```
[Stage 1: Nursery] ──> [Chapters 1-3: Fundamentals] ──> [Chapters 4-6: Control & Functions] ──> [Chapters 7-10: Advanced Mastery]
  • Python keywords      • Arithmetic operations         • Loops & comprehensions         • Object-Oriented design
  • Indentation & syntax • String transformations        • Recursion & closures           • Exception handling & I/O
  • Bracket balancing    • Lists & dictionaries          • Functional programming         • Algorithms & complexity
```

---

## 🛠️ Step-by-Step Teaching Workflows

### 1. Stage 1: The Developmental Nursery (Phonics & Syllables)
Run this when the robot is fresh or needs foundational reinforcement of basic Python syntax, keywords, and bracket pairing:

```powershell
# Run Stage 1 Nursery (teaches 30 basic phonics and structural patterns)
python run_stage1.py
```

*What to verify:*
- Verify that `Loss` decreases across epochs.
- Confirm that the phonics vocabulary (keywords like `def`, `return`, `if`, `for`, `in`) is recognized without tokenizer out-of-vocabulary errors.

---

### 2. Teaching a Specific Curriculum Chapter
Teach one focused topic at a time to build concrete algorithmic reasoning:

```powershell
# Chapter 1: Arithmetic & Math operations
python teach_robot.py --chapter 1

# Chapter 2: String manipulation & slicing
python teach_robot.py --chapter 2

# Chapter 3: Lists, sequences, & indexing
python teach_robot.py --chapter 3

# Chapter 4: Dictionaries & Key-Value mappings
python teach_robot.py --chapter 4

# Chapter 5: Loops, Iteration & Comprehensions
python teach_robot.py --chapter 5

# Chapter 6: Functions, Arguments & Recursion
python teach_robot.py --chapter 6

# Chapter 7: Object-Oriented Programming & Classes
python teach_robot.py --chapter 7

# Chapter 8: Exception Handling & Defensive Coding
python teach_robot.py --chapter 8

# Chapter 9: File I/O & Data Serialization
python teach_robot.py --chapter 9

# Chapter 10: Algorithms, Sorting & Complexity
python teach_robot.py --chapter 10
```

---

### 3. Interactive Pair-Teaching Console
To teach the robot interactively in real time with custom exercises:

```powershell
python teach_robot.py --interactive
```

In interactive mode:
1. Provide a prompt (e.g., `# Reverse a list\ndef reverse_list(items):`).
2. Code Robot attempts generation using its coarse-to-fine hierarchy.
3. The compiler verification loop analyzes syntax and returns AST diagnostics.
4. If an error occurs, provide the correction to update the curriculum memory.

---

### 4. Full Curriculum Sweep (Automated Classroom)
To teach the entire 10-chapter curriculum end-to-end:

```powershell
python teach_robot.py --all
```

*Expected Output:*
- Displays per-chapter lesson accuracy.
- Saves progress to `checkpoints/curriculum_progress.json`.

---

## 🔍 Validation & Diagnostic Checklist

After any teaching session, verify the robot's retention:

1. **AST Validity Check**:
   ```powershell
   python robot.py --prompt "# Check if number is prime\ndef is_prime(n):\n"
   ```
2. **Hybrid Consistency Realizer**:
   ```powershell
   python robot.py --hybrid --prompt "# Return sum of list\ndef sum_list(nums):\n"
   ```
3. Ensure the output shows:
   - `Chosen Source: Causal AR Head (Consistency-Guided)` or `NCM Parallel Fine Level`
   - `AST Syntax Valid : True`

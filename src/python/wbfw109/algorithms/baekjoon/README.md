
# Algorithms - backjoon

[![Solved.ac
Profile](http://mazassumnida.wtf/api/v2/generate_badge?boj=gnv112)](https://solved.ac/gnv112)

## Table of contents (TOC)

- [Algorithms - backjoon](#algorithms---backjoon)
  - [Table of contents (TOC)](#table-of-contents-toc)
  - [1. Link](#1-link)
  - [2. Main Directory Structure](#2-main-directory-structure)
  - [3. Analysis of problems](#3-analysis-of-problems)

## 1. Link

- [**My Profile - backjoon**](https://www.acmicpc.net/user/gnv112)
- [Help](https://help.acmicpc.net/)
  - [Tier Profile](https://github.com/mazassumnida/mazassumnida)
- [Homepage](https://www.acmicpc.net/)
  - [Tags](https://www.acmicpc.net/problem/tags)

## 2. Main Directory Structure

- All files can be tested by using command **"pytest \<file_name\>"**  
  - unittest: 🧪 pytest src/python/wbfw109/algorithms/baekjoon/gold.py::**\<test_function_name\>**  
    🛍️ e.g. ```pytest src/python/wbfw109/algorithms/baekjoon/gold.py::test_bundle_up_numbers```

- 📝 [gold.py](gold.py)

## 3. Analysis of problems

- Greedy (fundament)
  - find **rules** by debugging inductively.  
    - according to those rules, use proper **data structures**  
      - list, tuple, set  
      - stack, deque, heap
  - can function be terminated early (; **Early stopping**)?
  - simulation

- Sort
  - **design** elaborately loop that have exclusive range by using explicit pointers.
    - divide list into proper range in order to apply some rules.
  - Parametric search with Bisection method
  - 🛍️ e.g.
    - 3SUM

- Graph
  - variants of BFS
    - exploration with stop **by distance**.
    - use queue whose elements are **hashable** data.
    - data structure which represents explored points could be (**container** of boolean type **|** **set** type whose elements are hashable data).
  - Minimum spanning Tree

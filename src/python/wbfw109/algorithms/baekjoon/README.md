
# Algorithms - Baekjoon

[![Solved.ac
Profile](http://mazassumnida.wtf/api/v2/generate_badge?boj=gnv112)](https://solved.ac/gnv112)

---

Updated on 📅 2023-04-09 03:59:51

## Table of contents (TOC)

- [Algorithms - Baekjoon](#algorithms---baekjoon)
  - [Table of contents (TOC)](#table-of-contents-toc)
  - [1. External Link](#1-external-link)
  - [2. Main Directory Structure](#2-main-directory-structure)
  - [3. Analysis of problems](#3-analysis-of-problems)
  - [4. Issue tracking](#4-issue-tracking)
  - [5. Common Notation in solution files](#5-common-notation-in-solution-files)

## 1. External Link

- [**My Profile - Baekjoon**](https://www.acmicpc.net/user/gnv112)
- [Help](https://help.acmicpc.net/)
  - [Tier Profile](https://github.com/mazassumnida/mazassumnida)
  - [Comparisons Input speed](https://www.acmicpc.net/blog/view/56)
  - [Comparisons output speed](https://www.acmicpc.net/blog/view/57)
- [Homepage](https://www.acmicpc.net/)
  - [Tags](https://www.acmicpc.net/problem/tags)
  - **[Class](https://solved.ac/class)**

## 2. Main Directory Structure

- All files can be tested by using command **"pytest \<file_name\>"**  
  - unittest: 🧪 pytest src/python/wbfw109/algorithms/baekjoon/gold.py::**\<test_function_name\>**  
    🛍️ e.g. ```pytest src/python/wbfw109/algorithms/baekjoon/gold.py::test_bundle_up_numbers```

- 📝 [gold.py](gold.py) : 34 problems
- 📝 [platinum.py](platinum.py) : 1 problem

## 3. Analysis of problems

- Greedy (fundament) and **Dynamic programming** with **rolling window approach**
  - find **rules** by debugging inductively.  
    - according to those rules, use proper **data structures**.  
      - list, tuple, set  
      - stack, deque, heap (min heap, max heap)
    - consider immutable and mutable types.
  - can function be terminated early (; **Early stopping**)?  
    - class Exception can be used.

- Sorting
  - **design** elaborately loop that have exclusive range by using explicit pointers.
    - divide list into proper range in order to apply some rules.
- Simulations
  - BFS
    - BFS consists of mainly three components.
      - container (queue): **deque**, **heapq**, **set**
      - elements of container (route state): **point** with optional **\<essential objects\>**
      - trail: **boolean**, **boolean-like** (int; like visit counts, masking), **set**
    - to pause exploration **by depth**, and resume.
    - **Adjacency list** is useful when when represent undirected or directed graph.
    - 🛍️ e.g. (spread cells, distinguish zones from cells) in 2D matrix.
  - **Parametric search** with other simulations (This can be largely divided into Test algorithm and Decision algorithm (predicate)).
  - Binary search with custom Predicate
  - combinations

&nbsp; 
![#008000](https://placehold.co/15x15/008000/008000.png) &nbsp; **Named problems**
<details>
  <summary>🛍️ E.g.</summary>

- Knapsack
- Partial sum
- 3SUM
- Longest increasing subsequence

</details>

## 4. Issue tracking

- ✅ [Python Input Timeout from List multiplication rather than List comprehension](https://www.acmicpc.net/board/view/113417)

## 5. Common Notation in solution files

- in context of Time Complexity
  - O( BFS(**\<object\>**) ): Time complexity occurred in exploring **\<object\>** by using BFS; |V| + |E|
    - |V| is the number of vertices
    - |E| is the number of edges. if **\<object\>** is 2d array, it may be 4 directions.
- in context of Space Complexity
  - O( BFS(**\<object\>**) ): Space complexity to iterate **\<object\>** by using BFS; |V|

- in both contexts
  - N(\<object\>): the number of \<object\>

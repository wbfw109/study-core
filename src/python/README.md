# study-core-python

📝 about Utilities, see [root README.md](../../README.md#31-python-utilities)

## Table of contents (TOC)

- [study-core-python](#study-core-python)
  - [Table of contents (TOC)](#table-of-contents-toc)
  - [1. Algorithms](#1-algorithms)
    - [1.1 Basic](#11-basic)
    - [1.1.1 Optimization problems](#111-optimization-problems)
    - [1.2 Design patterns](#12-design-patterns)
    - [1.3 Specification](#13-specification)
  - [2. Libraries](#2-libraries)
  - [3. Snippet](#3-snippet)
  - [4. Toy program](#4-toy-program)

## 1. Algorithms

🅰️ Plan

1. to solve from lv 2 to lv 5 in Programmers site by Python3
2. to solve from lv 0 to lv 5 in Programmers site by SQL and Pandas
3. [goorm level](https://level.goorm.io/)
4. [Leet code](https://leetcode.com/)

![#008000](https://placehold.co/15x15/008000/008000.png) &nbsp; other **Problems & Solution**

- [**baekjoon**](wbfw109/algorithms/baekjoon/README.md)
- [**Programmers, Goorm**](wbfw109/algorithms/_practice/README.md)
- [unknown](wbfw109/algorithms/unknown/) ; Problems from Unknown source

  - 📝 [dp.py](wbfw109/algorithms/unknown/dp.py) ; (1) problems

**Main Tags** in use

- Optimization
  - Masking
  - Memoization
- Graph
  - BFS
- Dynamic programming
  - Rolling window approach
- Math
- Sequence
  - Subsequence

**Sub Tags** in use

- Sweep line algorithm
- Parametric search
  - Bisection method
- Backtracking
- \<Named problems\>
  - 3SUM variant

&nbsp;

---

![#f00](https://placehold.co/15x15/f00/f00.png) &nbsp; Summary : **[Available all output](https://wbfw109.github.io/visualization_manager/ipython_central_control.html)** from [3.3.1. Visualization Manager (Python)](../../README.md#331-visualization-manager-python)

### 1.1 Basic

| Type                         | Name                                             | location                                                                                                        |
| :--------------------------- | :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| Numerics                     | integer factorization                            | [class IntegerFactorization](wbfw109/labs/basics/numerics/factorization.py#L19)                                 |
| Numerics - definable numbers | decimal representation, positional notation      | [class DecimalRepresentation, PositionalNotation](wbfw109/labs/basics/numerics/rational_number.py#L19)          |
| Sequence Sorting             | bubble, quick, selection, heap, insertion, merge | [class ExchangeSorts, SelectionSorts, InsertionSorts, MergeSorts](wbfw109/labs/basics/sequences/sorting.py#L26) |
| Sequence Search              | binary                                           | [class BinarySearch](wbfw109/labs/basics/sequences/search.py#L47)                                               |
| Subsequence                  | longest increasing subsequence                   | [class LongestIncreasingSubsequence](wbfw109/labs/basics/sequences/subsequence.py#L21)                          |
| Substring                    | palindrome, Repeated substring                   | [class Palindrome, RepeatedSubstring](wbfw109/labs/basics/sequences/substrings.py#L21)                          |
| Graph basic                  | basics in terms of distance                      | [class GraphDistance](wbfw109/labs/basics/graphs/basic.py#L104)                                                 |
| Graph Search                 | DFS, BFS                                         | [class DfsAndBfs](wbfw109/labs/basics/graphs/search.py#L89)                                                     |
| Graph Routing                | Kruskal                                          | [class MinimumSpanningTree](wbfw109/labs/basics/graphs/routing.py#L102)                                         |

### 1.1.1 Optimization problems

| Type               | Name                                            | location                                                                                                                                       |
| :----------------- | :---------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| Named problems     | 3SUM, Subset Sum, Travelling Salesman, Knapsack | [class ThreeSumProblem, SubsetSumProblem, TravellingSalesmanProblem, KnapsackProblem](wbfw109/labs/basics/optimizations/named_problems.py#L48) |
| Linear programming | Set cover problem                               | [class SetCoverProblem](wbfw109/labs/basics/optimizations/linear_programming.py#L57)                                                           |

&nbsp;

---

### 1.2 Design patterns

| Type               | Name      | location                                               |
| :----------------- | :-------- | :----------------------------------------------------- |
| Creational pattern | interning | [class Interning](wbfw109/labs/design_patterns.py#L18) |

&nbsp;

---

### 1.3 Specification

| Type                     | Name                                                                | location                                                                                                                       |
| :----------------------- | :------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------- |
| Built-in code constructs | unpacking variables, for-in loop, operators, lambda, iterators      | [class UnpackingVariables, ForStatement, Operators, LambdaExpression, Iterators](wbfw109/labs/builtins/code_constructs.py#L24) |
| Built-in functions       | format, min, range, sort, zip, Sum-prod and min-max                 | [class FormatFunc, MinFunc, RangeFunc, SortFunc, ZinFunc, SumProdAndMinMax](wbfw109/labs/builtins/functions.py#L18)            |
| Built-in system          | CPython advantage, array in memory, data size, zero-based numbering | [class CPythonAdvantage, ArrayInMemory DataSize, ZeroBasedNumbering](wbfw109/labs/builtins/system.py#L18)                      |
| Dunders names            | Truthy values, getattr, iter                                        | [class DundersTruthyValue, DundersGetattr, DundersIter](wbfw109/labs/dunders_names.py#L36)                                     |
| **Data types**           | numerics (int), sequence (list), mapping (dict)                     | [class IntDT, SequenceDT, ListDT, DictDT](wbfw109/labs/basics/data_stucture.py#L28)                                            |

🧊 Some file may have profiling to some pieces. These code are actually not run from Visualization Manager, but will show "**Profile conclusion**".

---

## 2. Libraries

- objects.[object](wbfw109/libs/objects/object.py) ; Operations on first-class object.
  - About Variables 🔪 default value map, get field annotations, initialize_fields
  - About Nested class, Inheritance 🔪 get (child, descendant) classes, is inner class, get outer class
  - About Converting json to dataclass 🔪 SelfReferenceHelper, MixInJsonSerializable
- [parsing](wbfw109/libs/parsing.py) ; Parsing trees.
  - About Implicit node 🔪 convert Implicit syntax node to Syntax Tree (recursive dict)
  - About Third-party compatibility 🔪 convert syntax Tree to Recursive Tuple tree (library: svgling to draw Syntax tree as .svg image format)
- [path](wbfw109/libs/path.py) ; Path resolutions.
  - About module format string 🔪 get valid path pair from sys path, convert module path to qualified name, get module name list from paths
- [string](wbfw109/libs/string.py) ; String operations.
  - About naming convention 🔪 rename to snake case, rename to snake case with replace
- [typing](wbfw109/libs/typing.py) ; Typing for common used type.
  - About Generic TypeVar 🔪 T, DST (Data Structure type decorated with dataclasses.dataclass)
  - About Generic with Typed 🔪 Json Primitive ValueType, Json Type, Recursive Tuple, Single Linked List

---

## 3. [Snippet](wbfw109/template/snippet.py)

- **Transpose iterations**
  - get_square_iterators_against_gravity()
  - assemble(), reassemble(), reassemble_i()

---

## 4. Toy program

| Language | Type             | Main tech            | Name                                      | location                                                                               |
| :------- | :--------------- | :------------------- | :---------------------------------------- | :------------------------------------------------------------------------------------- |
| Python   | Crawling         | XPath                | Google CodeJam Crawler                    | [function crawl_algorithms_code_jam()](wbfw109/libs/utilities/self/algorithms.py#L200) |
| Python   | Image Processing | Generator, decorator | Thumbnail Generator                       | [class ThumbGenExample](wbfw109/labs/dunders_names.py#L411)                            |
| Python   | Communication    | socket, selector     | Echo: (Stream type, muxing, Non-blocking) | [class SockEchoCommunication](wbfw109/labs/networking.py#L73)                          |


# study-core-python

📝 about Utilities, see [root README.md](../../README.md#31-python-utilities)

## Table of contents (TOC)

- [study-core-python](#study-core-python)
  - [Table of contents (TOC)](#table-of-contents-toc)
  - [1. Algorithms](#1-algorithms)
    - [1.1 Basic](#11-basic)
    - [1.2 Design patterns](#12-design-patterns)
    - [1.3 Specification](#13-specification)
  - [2. Libraries](#2-libraries)
  - [3. Toy program](#3-toy-program)

## 1. Algorithms

![#008000](https://placehold.co/15x15/008000/008000.png) &nbsp; other **Problems & Solution**

- [**baekjoon**](wbfw109/algorithms/baekjoon/README.md)

---

![#f00](https://placehold.co/15x15/f00/f00.png) &nbsp; Summary : **[Available all output](https://wbfw109.github.io/visualization_manager/ipython_central_control.html)** from [3.3.1. Visualization Manager (Python)](../../README.md#331-visualization-manager-python)

### 1.1 Basic

|Type|Name|location|
|:---|:---|:---|
|Sorting  |bubble, quick, selection, heap, insertion, merge |[class ExchangeSorts, SelectionSorts, InsertionSorts, MergeSorts](wbfw109/labs/basics/sequences/sorting.py#L157)|
|Graph search  |DFS, BFS |[class DfsAndBfs](wbfw109/labs/basics/graphs/search.py#L142)|
|Graph routing  |(Minimum Spanning Tree) |[class Kruskal](wbfw109/labs/basics/graphs/routing.py#L160)|

&nbsp;

---

### 1.2 Design patterns

|Type|Name|location|
|:---|:---|:---|
|Creational pattern  |interning |[class Interning](wbfw109/labs/design_patterns.py#L72)|

&nbsp;

---

### 1.3 Specification

|Name|location|
|:---|:---|
Built-ins statements: (for-in loop)  |[class ForStatement, Operators](wbfw109/labs/builtins/statements.py#L74)|
Built-ins expressions: (lambda)  |[class LambdaExpression](wbfw109/labs/builtins/expressions.py#L74)|
Built-ins functions: (zip(), min())  |[class ZipFunc, MinFunc](wbfw109/labs/builtins/functions.py#L71)|
Built-ins system: (Zero Based Numbering)  |[class DataSize, ZeroBasedNumbering](wbfw109/labs/builtins/system.py#L74)|
Dunders names: (Truthy values, getattr, iter)  |[class DundersTruthyValue, DundersGetattr, DundersIter](wbfw109/labs/dunders_names.py#L74)|

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

## 3. Toy program

|Language|Type|Main tech|Name|location|
|:---|:---|:---|:---|:---|
|Python  |Crawling  |XPath  |Google CodeJam Crawler  |[function crawl_algorithms_code_jam()](wbfw109/libs/utilities/self/algorithms.py#L200)|
|Python  |Image Processing  |Generator, decorator  |Thumbnail Generator  |[class ThumbGenExample](wbfw109/labs/dunders_names.py#L411)|
|Python  |Communication  |socket, selector  |Echo: (Stream type, muxing, Non-blocking) |[class SockEchoCommunication](wbfw109/labs/networking.py#L73)|

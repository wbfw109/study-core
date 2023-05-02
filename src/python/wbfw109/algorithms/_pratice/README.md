# Algorithms - practice

---

Updated on 📅 2023-05-01 06:40:36

## Table of contents (TOC)

- [Algorithms - practice](#algorithms---practice)
  - [Table of contents (TOC)](#table-of-contents-toc)
  - [1. Practice sites](#1-practice-sites)
    - [1.1 Programmers](#11-programmers)
      - [1.1.1 External Link](#111-external-link)
      - [1.1.2 Main Directory Structure](#112-main-directory-structure)
    - [1.2 Goorm](#12-goorm)
      - [1.2.1 External Link](#121-external-link)
      - [1.2.2 Main Directory Structure](#122-main-directory-structure)
  - [2. Analysis for fast writing code in given time](#2-analysis-for-fast-writing-code-in-given-time)

## 1. Practice sites

- **Purpose**
  - to practice in an environment similar to the actual coding test, rather than learning new theory (Baekjoon is used for latter purpose)
    - so codes may be very simple because I have to solve a problem in given time.
    - and I've written **tags** to almost problems.
  - to understand completely problem and **match contents to proper underlying algorithms.**

### 1.1 [Programmers](programmers/)

#### 1.1.1 External Link

- [**My Profile - Programmers**](https://career.programmers.co.kr/pr/gnv112_5261)
- Help
  - Compile options in a problem tab
    - Python3: Python 3.8.5
- [Problems](https://school.programmers.co.kr/learn/challenges)
  - [Skill checks](https://career.programmers.co.kr/skill_checks)

#### 1.1.2 Main Directory Structure

- 📝 [lv0.py](programmers/lv0.py) ; (224 / 224) problems
- 📝 [lv1.py](programmers/lv1.py) ; (5 / 77) problems

### 1.2 [Goorm](goorm/)

#### 1.2.1 External Link

- Help
  - Compile options in a problem tab
    - Python3: Python3.9 (3.9.6)
    - Runtime Error includes "Memory limit exceeded". (in custom test cases, "Error 137" occurs)
- [Problems](https://level.goorm.io/)

#### 1.2.2 Main Directory Structure

- 📝 [lv1.py](goorm/lv1.py) ; (51 / 74-1) problems
  - "-1" is problems that can not solved by using Python; id: 151664
- 📝 [lv4.py](goorm/lv4.py) ; (0+1 / 27+1) problems
  - "+1" is from "[Goorm devth (Depth of Developer) testing page](https://devth.goorm.io/)"
- 📝 [lv5.py](goorm/lv5.py) ; (1 / 11) problems

## 2. Analysis for fast writing code in given time

- Name informally variables as simple **acronym**.
  - 🛍️ e.g.
    - dq := dequeue
    - ev := explored_v
    - nv := neighbor_v
    - p := point, pointer
    - dx, dy := direction for x, y
    - s, l, word := string, letter, word
    - q, r = quotient, remainder
- If imported functions name with package name is long,
  - **import directly functions** instead o importing a package by using "from" and "import" keyword.
- Don't try to force the use of Type hints.

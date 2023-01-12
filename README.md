
# study-core

📝 **Note that** this repository is currently integrating from [outdated small project sources](src/python/wbfw109/outdated/). so can be incomplete.

- [study-core](#study-core)
  - [1. Installation](#1-installation)
    - [1.1. Main Directory Structure](#11-main-directory-structure)
    - [1.2. Steps to start](#12-steps-to-start)
      - [1.2.1. Common settings](#121-common-settings)
      - [1.2.2. Settings by workspace](#122-settings-by-workspace)
  - [2. DevOps Toolchains as GitOps](#2-devops-toolchains-as-gitops)
    - [2.1. Version Control: GitHub](#21-version-control-github)
    - [2.2. Pipeline: GitHub Actions](#22-pipeline-github-actions)
      - [2.2.1 Workflow: CD to GitHub Pages](#221-workflow-cd-to-github-pages)
    - [2.3. Packaging: Docker](#23-packaging-docker)
  - [3. Sources](#3-sources)
    - [3.1. Basic algorithms](#31-basic-algorithms)
      - [3.1.1 Python: Specification](#311-python-specification)
    - [3.2. Libraries](#32-libraries)
      - [3.2.1. Python Libraries](#321-python-libraries)
    - [3.3. Utilities](#33-utilities)
      - [3.3.1. Python Utilities](#331-python-utilities)
        - [3.3.1. Visualization Manager (Python)](#331-visualization-manager-python)
      - [3.3.2. Shell Utilities](#332-shell-utilities)
        - [3.3.2.1. Backup Docker volumes (Bash)](#3321-backup-docker-volumes-bash)
  - [4. Toy program](#4-toy-program)
  - [5. Services](#5-services)
    - [5.1. Glossary service](#51-glossary-service)
    - [5.2. E-Commerce Service](#52-e-commerce-service)

## 1. Installation

### 1.1. Main Directory Structure

- 📁 features ; for BDD **_(incomplete...)_**
- 📁 [ref](ref/) ; crawled data, project settings (toml), etc.
- 📁 [src/bash/wbfw109](src/bash/wbfw109/) ; common scripts like (setup.sh, install-selenium-driver.sh) and custom scripts.
- 📁 [src/python/wbfw109](src/python/wbfw109/)
  - 📁 algorithms ; solutions in coding competitions **_(incomplete...)_**
  - 📁 [labs](src/python/wbfw109/labs/) ; research on specification
  - 📁 tutorials ; as the word itself. when starting a library.
  - 📁 [libs](src/python/wbfw109/libs/) ; common libraries
    - 📁 [utilities](src/python/wbfw109/libs/utilities)
      - 📁 [self](src/python/wbfw109/libs/utilities/self/) ; code that should only be used in this project: crawling, settings
      - 📑 ipython.py ; visualization helper in Python Interactive Windows
    - 📁 [objects](src/python/wbfw109/libs/objects/)
    - 📑~ typing.py, path.py, string.py, parsing.py
- 📁 src/typescript
- 📁 [services](services/) ; glossary_service, e_commerce_service (in development)

### 1.2. Steps to start

#### 1.2.1. Common settings

- VS Code
  1. download [VS Code](https://code.visualstudio.com/download)
  2. download [some extensions](.vscode/extensions.json)

- Python
  1. download [poetry](https://python-poetry.org/docs/#installation).

- Typescript
    1. download [yarn](https://yarnpkg.com/getting-started/install) (yarn berry PnP).
    2. download [ZipFS](https://marketplace.visualstudio.com/items?itemName=arcanis.vscode-zipfs) in VS Code Extensions from [Editor support](https://yarnpkg.com/getting-started/migration#editor-support).  
      and run command: ```yarn dlx @yarnpkg/sdks vscode```  
      and select [Use Workspace Version](https://code.visualstudio.com/docs/typescript/typescript-compiling#_using-the-workspace-version-of-typescript) in VSCode

- Docker
    1. download [Docker](https://docs.docker.com/engine/install/).

#### 1.2.2. Settings by workspace

- study-core (repository)  
  In root directory of the repository,  
  - Python
    - run command: ```poetry install --with web,db,vision && poetry env info```
    - run command (F1) in VSCode: ```>Python: Select Interpreter``` a path from upper output ("Virtualenv - Executable")
  - Typescript: run command: ```yarn install```
  - _**(Optional)**_ Docker: run command: ```docker compose -f ref/dockerfiles/docker-compose.yml up -d```  
    - [docker-comose.yml](ref/dockerfiles/docker-compose.yml)
      - **Plant UML** in order to write UML diagram
      - **Mongo DB** (Document based NoSQL) with **Mongo Express**
      - **PostgreSQL** (R-DBMS) with **pg Admin**

- glossary_service (service)  
  In the [service directory](services/glossary_service/),  
  - Python
    - run command: ```poetry install && poetry env info```
    - run command (F1) in VSCode: ```>Python: Select Interpreter``` a path from upper output ("Virtualenv - Executable")

- e_commerce_service (service)  
  📰 In development: Python, Typescript, Docker

&nbsp;

---

## 2. DevOps Toolchains as GitOps

### 2.1. Version Control: GitHub

It is Single contributor project.  
**Branching strategy**

- main
- dev (features, bugfix, hotfix)

### 2.2. Pipeline: GitHub Actions

The main reason I choose this:

- [About billing for GitHub Actions](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#about-billing-for-github-actions)  
    > GitHub Actions usage is free for standard GitHub-hosted runners in public repositories, and for self-hosted runners.

#### 2.2.1 [Workflow: CD to GitHub Pages](.github/workflows/github-pages-CD-workflow.yml)

It is triggered in only main branch.

When [3.3.1. Visualization Manager (Python)](#331-visualization-manager-python) outputs a .html file,
ipython_central_control.html file will be modified. So whenever only the file changes and push operation is occurred, this workflow triggered.

- It not traces all file in GitHub Pages repository when pull & push.  
Because a part of directory in GitHub Pages repository may be later written, so I used sparse-checkout in Git SCM.

### 2.3. Packaging: Docker

Currently, one [docker-comose.yml](ref/dockerfiles/docker-compose.yml) exists used in development environment.

&nbsp;

---

## 3. Sources

### 3.1. Basic algorithms

![#f00](https://placehold.co/15x15/f00/f00.png) &nbsp; Summary : **[Available all output](https://wbfw109.github.io/visualization_manager/ipython_central_control.html)** from [3.3.1. Visualization Manager (Python)](#331-visualization-manager-python)

|Language|Type|Name|location|
|:---|:---|:---|:---|
|Python  |Sorting  |bubble, quick, selection, heap, insertion, merge |[class ExchangeSorts, SelectionSorts, InsertionSorts, MergeSorts](src/python/wbfw109/labs/basics/sequences/sorting.py#L158)|
|Python  |Graph search  |DFS, BFS |[class DfsAndBfs](src/python/wbfw109/labs/basics/graphs/search.py#L142)|

---

#### 3.1.1 Python: Specification

|Name|location|
|:---|:---|
Built-ins statements: (for-in loop)  |[class ForStatement](src/python/wbfw109/labs/builtins/statements.py#L74)|
Built-ins expressions: (lambda)  |[class LambdaExpression](src/python/wbfw109/labs/builtins/expressions.py#L74)|
Built-ins functions: (zip(), min())  |[class ZipFunc, MinFunc](src/python/wbfw109/labs/builtins/functions.py#L71)|
Built-ins system: (Zero Based Numbering)  |[class ZeroBasedNumbering](src/python/wbfw109/labs/builtins/system.py#L74)|
Dunders names: (Truthy values, getattr, iter)  |[class DundersTruthyValue, DundersGetattr, DundersIter](src/python/wbfw109/labs/dunders_names.py#L74)|

---

### 3.2. Libraries

#### 3.2.1. Python Libraries

- objects.[object](src/python/wbfw109/libs/objects/object.py) ; Operations on first-class object.
  - About Variables 🔪 default value map, get field annotations, initialize_fields
  - About Nested class, Inheritance 🔪 get (child, descendant) classes, is inner class, get outer class
  - About Converting json to dataclass 🔪 SelfReferenceHelper, MixInJsonSerializable
- [parsing](src/python/wbfw109/libs/parsing.py) ; Parsing trees.
  - About Implicit node 🔪 convert Implicit syntax node to Syntax Tree (recursive dict)
  - About Third-party compatibility 🔪 convert syntax Tree to Recursive Tuple tree (library: svgling to draw Syntax tree as .svg image format)
- [path](src/python/wbfw109/libs/path.py) ; Path resolutions.
  - About module format string 🔪 get valid path pair from sys path, convert module path to qualified name, get module name list from paths
- [string](src/python/wbfw109/libs/string.py) ; String operations.
  - About naming convention 🔪 rename to snake case, rename to snake case with replace
- [typing](src/python/wbfw109/libs/typing.py) ; Typing for common used type.
  - About Generic TypeVar 🔪 T, DST (Data Structure type decorated with dataclasses.dataclass)
  - About Generic with Typed 🔪 Json Primitive ValueType, Json Type, Recursive Tuple, Single Linked List

### 3.3. Utilities

#### 3.3.1. Python Utilities

##### 3.3.1. [Visualization Manager (Python)](src/python/wbfw109/libs/utilities/ipython.py#L389)

|Language|Type|Main tech|
|:---|:---|:---|
|Python  |Visualization  |pandas, IPython, Generic, inspect, Path, contextlib  |

It raises readability of contents of classes that inherits \<VisualizationRoot\> | \<MixInParentAlgorithmVisualization\>, \<ChildAlgorithmVisualization\> **in environment using iPython (Python Interactive Windows).**

- In the case of \<VisualizationRoot\>, You can use it simply by implementing these: \_\_init\_\_(self), \_\_str\_\_(self), test_case(cls).  
  Just call the class method \<call_root_classes\>.
- In a case where to group classes or detail implementation is required, you can use \<MixInParentAlgorithmVisualization\>, \<ChildAlgorithmVisualization\> such as in solutions of sorting.  
  Also you are able to pass a data object as same data by calling class method \<call_parent_algorithm_classes\> for benchmarking.  
  For example in sorting solutions, \<target_list\> was passed as argument.

You could select classes you want to show by passing argument \<only_class_list>\, otherwise it shows available classes in a module.

It is valid on consecutively called situations between each different files in order to display various contents into one Interactive Windows.
By tracing call stack, it only shows contents of current running file or cell rather than the modules that have been executed so far.

Moreover you can manipulate through central control by calling class method \<call_modules\> with searchable paths argument as (absolute | relative) (packages | modules (.py)). It automatically filters same absolute path and selects one of shortest relative path.

Most of my implementations comply this rule. If you want to check all contents that have been implemented in this way so far, just open ![#f00](https://placehold.co/15x15/f00/f00.png) **[Available all output](https://wbfw109.github.io/visualization_manager/ipython_central_control.html).**

- or you could directly run.
    1. Open file: [labs_visualization.py](src/python/wbfw109/central_control/labs_visualization.py)  
    2. Run the file by typing ```Ctrl+Enter (command: Jupyter: Run Current Cell)``` with nothing changes.  

👍 Furthermore [in this module](src/python/wbfw109/libs/utilities/ipython.py), you could also visualize graph as .svg image format by using function \<visualize_implicit_tree\>

![#008000](https://placehold.co/15x15/008000/008000.png) &nbsp; **Pictures**
<details>
  <summary>Examples (consecutively calling different files)</summary>
  
![Visualization Manager 1](resources_readme/visualization_manager/1.png?raw=1)
![Visualization Manager 2](resources_readme/visualization_manager/2.png?raw=1)
![Visualization Manager 3](resources_readme/visualization_manager/3.png?raw=1)
</details>

&nbsp;

---

#### 3.3.2. Shell Utilities

##### 3.3.2.1. [Backup Docker volumes (Bash)](src/bash/wbfw109/docker-backup-volumes.sh)

|Language|Type|Main tech|
|:---|:---|:---|
|Bash  |Recovery  |Docker  |

If you pass volume names, it filters unknown volumes and backups remainder with suffix in ISO-8601 datetime format in which colons (:) are replaced with "".

Run ```src/bash/wbfw109/docker-backup-volumes.sh <volume_name_1> [<volume_name_2>, ...]```.  
Backed up files will be stored in ```ref/dockerfiles/backup```. (hardcoded)

&nbsp;

---

## 4. Toy program

|Language|Type|Main tech|Name|location|
|:---|:---|:---|:---|:---|
|Python  |Crawling  |XPath  |Google CodeJam Crawler  |[function crawl_algorithms_code_jam()](src/python/wbfw109/libs/utilities/self/algorithms.py#L200)|
|Python  |Image Processing  |Generator, decorator  |Thumbnail Generator  |[class ThumbGenExample](src/python/wbfw109/labs/dunders_names.py#L411)|
|Python  |Communication  |socket, selector  |Echo: (Stream type, muxing, Non-blocking) |[class SockEchoCommunication](src/python/wbfw109/labs/networking.py#L73)|

---

## 5. Services

Purpose of All Services is **Scaffolding** or **Minimum viable product (MVP)** for PoC

### 5.1. Glossary service

It provides English words with description in Korean related with Computer Science as **tree structure**, that I learned  .

Main tech is **Pynecone** (Web full stack framework); Set of **_FastAPI, NextJS, React_**

- Run command: ```pc init``` (one time) and ```pc run --env prod```

[Words Data](ref/computer_science_words_korean.json) and [**Entry point**: glossary_app.py](services/glossary_service/src/wbfw109/glossary_app/glossary_app/glossary_app.py)

- [1-Plan](services/glossary_service/devops/stages/1-plan.md)
- [2-Create](services/glossary_service/devops/stages/2-create.md)
- [3-Verify](services/glossary_service/devops/stages/3-verify.md)
- [9-Promotion](services/glossary_service/devops/stages/9-promotion.md)

![#008000](https://placehold.co/15x15/008000/008000.png) &nbsp; **Video**

[![Glossary app PV](https://img.youtube.com/vi/LBqgitY_j5A/0.jpg)](https://youtu.be/LBqgitY_j5A "Glossary app PV")

### 5.2. E-Commerce Service

📰 Currently in development

- [1-Plan (Draft)](services/e_commerce_service/devops/stages/1-plan.md)  
  - [Deployment diagram](services/e_commerce_service/resources_readme/diagrams/deployment.svg?raw=1)
- 3-Verify
  - coverages
    - backend
    <!-- - [backend](https://wbfw109.github.io/coverages/backend/index.html) -->

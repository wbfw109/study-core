# study-core

📝 See each docs about Algorithms, Libraries, Toy program.

- [study-core-python](src/python/README.md)

💫 This repository is currently integrating from [outdated small project sources](outdated/). so can be incomplete.

## Table of contents (TOC)

- [study-core](#study-core)
  - [Table of contents (TOC)](#table-of-contents-toc)
  - [1. Installation](#1-installation)
    - [1.1. Main Directory Structure](#11-main-directory-structure)
    - [1.2. Steps to start](#12-steps-to-start)
      - [1.2.1. Common settings](#121-common-settings)
      - [1.2.2. Settings by workspace](#122-settings-by-workspace)
  - [2. DevOps Toolchains as GitOps](#2-devops-toolchains-as-gitops)
    - [2.1. Version Control: GitHub](#21-version-control-github)
    - [2.2. Automation of jobs: tasks.json (in VS code)](#22-automation-of-jobs-tasksjson-in-vs-code)
    - [2.3. Pipeline: GitHub Actions](#23-pipeline-github-actions)
    - [2.4. Packaging: Docker](#24-packaging-docker)
  - [3. Utilities](#3-utilities)
    - [3.1. Python Utilities](#31-python-utilities)
      - [3.3.1. Visualization Manager (Python)](#331-visualization-manager-python)
    - [3.2. Shell Utilities](#32-shell-utilities)
      - [3.2.1. install Protocol Buffers Compiler (protoc) 3 (Bash)](#321-install-protocol-buffers-compiler-protoc-3-bash)
      - [3.2.2. Backup Docker volumes (Bash)](#322-backup-docker-volumes-bash)
  - [4. Services](#4-services)
    - [4.1. Glossary service](#41-glossary-service)
    - [4.2. E-Commerce Service](#42-e-commerce-service)
      - [4.2.1. Pipeline: GitHub Actions](#421-pipeline-github-actions)

## 1. Installation

### 1.1. Main Directory Structure

<!--
- 📁 docs ; sub-docs **_(incomplete...)_**
- 📁 features ; for BDD **_(incomplete...)_** -->

- 📁 [environment](environment/) ; common settings for OS (WSL 2)
- 📁 [ref](ref/) ; crawled data, project settings (toml), etc.
- 📁 [src/bash/wbfw109/utilities](src/bash/wbfw109/utilities/)
  - 📁 [setup](src/bash/wbfw109/utilities/setup/)
  - 📁 [self/tasks](src/bash/wbfw109/utilities/self/tasks/) ; scripts used in .vscode/tasks.json
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

- [WSL 2 settings](environment/README.md)

- _IDE_: VS Code  
  🔱 _Main reason why I choose this_: [Lightweight and powerful.](https://code.visualstudio.com/docs/setup/setup-overview)

  1. download [VS Code](https://code.visualstudio.com/download)
     1. download [some extensions](.vscode/extensions.json)

- Python

  1. download latest [Python directly](https://www.python.org/downloads/) or using [pyenv](https://github.com/pyenv/pyenv#installation) (Recommend)
  2. download latest [poetry](https://python-poetry.org/docs/#installation).  
     🔱 _Main reason why I choose this_: [At that time](https://github.com/wbfw109/crawling_copy#1-select-python-virtual-environment), time in resolving dependencies was faster than Pipenv.

- Typescript  
  🔱 _Main reason why I choose this_: [Static type-checking](https://www.typescriptlang.org/docs/handbook/2/basic-types.html)

  1. download [yarn](https://yarnpkg.com/getting-started/install) (yarn berry PnP).  
     🔱 _Main reason why I choose this_: [Plug'n'Play](https://yarnpkg.com/features/pnp)  
     &nbsp;
     1. download [ZipFS](https://marketplace.visualstudio.com/items?itemName=arcanis.vscode-zipfs) in VS Code Extensions from [Yarn Editor support](https://yarnpkg.com/getting-started/migration#editor-support).
     2. run command: `yarn dlx @yarnpkg/sdks vscode`
     3. select [Use Workspace Version](https://code.visualstudio.com/docs/typescript/typescript-compiling#_using-the-workspace-version-of-typescript) in VSCode

- Docker

  1. download [Docker](https://docs.docker.com/engine/install/).

- Protocol Buffers  
  🔱 _Main reason why I choose this_: [Communication between internal microservices as SsoT](https://cloud.google.com/run/docs/triggering/grpc)

  1. run file: [Protocol Buffers Compiler (protoc)](src/bash/wbfw109/utilities/setup/install_protoc_3.sh) **(only for linux-x86_64 OS).** or follow [the guide](https://github.com/protocolbuffers/protobuf#protocol-compiler-installation)

#### 1.2.2. Settings by workspace

---

- study-core (repository)  
  In root directory of the repository,

  - Python

    1. run command: `poetry install --with web,db,vision,test,dev && poetry env info`  
       _**(Optional)**_ packages for development: ... `--with test,dev`

    2. run command (F1) in VSCode: `>Python: Select Interpreter` a path from upper output ("Virtualenv - Executable")

  - Typescript: run command: `yarn install`

  - _**(Optional)**_ Docker: run command: `docker compose -f docker/docker-compose.yml up -d`
    - [docker-comose.yml](docker/docker-compose.yml)
      - **Plant UML** in order to write UML diagram
      - **Mongo DB** (Document based NoSQL) with **Mongo Express**
      - **PostgreSQL** (R-DBMS) with **pg Admin**

---

- glossary_service (service)  
  In the [glossary service directory](services/glossary_service/),  
  &nbsp;

  - Python
    1. run command: `poetry install && poetry env info`
    2. run command (F1) in VSCode: `>Python: Select Interpreter` a path from upper output ("Virtualenv - Executable")

---

📰 In development

- e_commerce_service (service)  
  In the [e-commerce service directory](services/e_commerce_service/),  
  &nbsp;

  - Python

    1. run command: `poetry install --with test,dev && poetry env info`  
       _**(Optional)**_ packages for development: ... `--with test,dev`
    2. run command (F1) in VSCode: `>Python: Select Interpreter` a path from upper output ("Virtualenv - Executable")

  - Protocol Buffers

&nbsp;

---

## 2. DevOps Toolchains as GitOps

📝 About other elements for services, see the corresponding service header.

study-core

- [3-Verify](devops/stages/3-verify.md)

### 2.1. Version Control: GitHub

It is Single contributor project so I set Lock branch in Branch protection rule about collaborators.

- **Branching strategy**
  - main
  - dev (features, bugfix, hotfix)

### 2.2. Automation of jobs: [tasks.json](.vscode/tasks.json) (in VS code)

- **List of jobs**
  - git (checkout, add, commit, rebase, merge, push) on (dev | main) branch
  - Run web server for tutorials: (Fastapi, Svelte)

### 2.3. Pipeline: GitHub Actions

🔱 _Main reason why I choose this_: [About billing for GitHub Actions](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#about-billing-for-github-actions)

> GitHub Actions usage is free for standard GitHub-hosted runners in public repositories, and for self-hosted runners.

- [Workflow: CD to GitHub Pages](.github/workflows/github_pages-CD.yml)  
  Main tech is **git sparse-checkout**.  
  This workflow triggered whenever
  1. [3.3.1. Visualization Manager (Python)](#331-visualization-manager-python) outputs a .html file (ipython_central_control.html) and modified,
  2. and if push operation is occurred in only main branch.

### 2.4. Packaging: Docker

Currently, one [docker-comose.yml](docker/docker-compose.yml) exists used in development environment.

&nbsp;

---

## 3. Utilities

### 3.1. Python Utilities

#### 3.3.1. [Visualization Manager (Python)](src/python/wbfw109/libs/utilities/ipython.py#L389)

| Language | Type          | Main tech                                           |
| :------- | :------------ | :-------------------------------------------------- |
| Python   | Visualization | pandas, IPython, Generic, inspect, Path, contextlib |

![#008000](https://placehold.co/15x15/008000/008000.png) &nbsp; **Pictures**

<details>
  <summary>🛍️ E.g. (consecutively calling different files)</summary>
  
![Visualization Manager 1](resources_readme/visualization_manager/1.png?raw=1)
![Visualization Manager 2](resources_readme/visualization_manager/2.png?raw=1)
![Visualization Manager 3](resources_readme/visualization_manager/3.png?raw=1)
</details>

![#f00](https://placehold.co/15x15/f00/f00.png) **[Available all output](https://wbfw109.github.io/visualization_manager/ipython_central_control.html)**
; check all contents that have been implemented in this way so far

It raises readability of contents of classes that inherits \<VisualizationRoot\> | \<MixInParentAlgorithmVisualization\>, \<ChildAlgorithmVisualization\> **in environment using iPython (Python Interactive Windows).**

- In the case of \<VisualizationRoot\>, You can use it simply by implementing these: \_\_init\_\_(self), \_\_str\_\_(self), test_case(cls).  
  Just call the class method \<call_root_classes\>.
- In a case where to group classes or detail implementation is required, you can use \<MixInParentAlgorithmVisualization\>, \<ChildAlgorithmVisualization\> such as in solutions of sorting.  
  Also you are able to pass a data object as same data by calling class method \<call_parent_algorithm_classes\> for benchmarking.  
  🛍️ E.g. in sorting solutions, \<target_list\> was passed as argument.

You could select classes you want to show by passing argument \<only_class_list>\, otherwise it shows available classes in a module.

It is valid on consecutively called situations between each different files in order to display various contents into one Interactive Windows.
By tracing call stack, it only shows contents of current running file or cell rather than the modules that have been executed so far.

Moreover you can manipulate through central control by calling class method \<call_modules\> with searchable paths argument as (absolute | relative) (packages | modules (.py)). It automatically filters same absolute path and selects one of shortest relative path.

👍 Furthermore [in this module](src/python/wbfw109/libs/utilities/ipython.py), you could also visualize graph as .svg image format by using function \<visualize_implicit_tree\>

&nbsp;

---

### 3.2. Shell Utilities

#### 3.2.1. [install Protocol Buffers Compiler (protoc) 3 (Bash)](src/bash/wbfw109/utilities/setup/install_protoc_3.sh)

| Language | Type  | Main tech              |
| :------- | :---- | :--------------------- |
| Bash     | Setup | Github API, jq, semver |

- 🔎 Usage: `install_protoc_3.sh [-u | --update]`.

Install protoc 3 as linux-x86_64 distribution from [GitHub Releases](https://github.com/protocolbuffers/protobuf).  
"jq" parses json data received from GitHub API.

If protoc already installed, it compares current version with latest version and prints whether current protoc is up to date or not.

- and if with --update arguments, update.

#### 3.2.2. [Backup Docker volumes (Bash)](src/bash/wbfw109/utilities/docker-backup-volumes.sh)

| Language | Type     | Main tech |
| :------- | :------- | :-------- |
| Bash     | Recovery | Docker    |

- 🔎 Usage: `docker-backup-volumes.sh <volume_name_1> [<volume_name_2> ...]`.

If you pass volume names, it filters unknown volumes and backups remainder with suffix in ISO-8601 datetime format in which colons (:) are replaced with "".

Backed up files will be stored in `docker/volume_backup/`. (hardcoded)

&nbsp;

---

## 4. Services

Purpose of All Services is **Scaffolding** or **Minimum viable product (MVP)** for PoC

### 4.1. Glossary service

It provides English words with description in Korean related with Computer Science as **tree structure**, that I learned .

Main tech is **Pynecone** (Web full stack framework); Set of **_FastAPI, NextJS, React_**

How to run:

- In [glossary_app directory](services/glossary_service/src/wbfw109/glossary_app),  
  Run command: `pc init` (one time) and `pc run --env prod`

[Words Data](ref/computer_science_words_korean.json) and [**Entry point**: glossary_app.py](services/glossary_service/src/wbfw109/glossary_app/glossary_app/glossary_app.py)

- [1-Plan](services/glossary_service/devops/stages/1-plan.md)
- [2-Create](services/glossary_service/devops/stages/2-create.md)
- [3-Verify](services/glossary_service/devops/stages/3-verify.md)
- [9-Promotion](services/glossary_service/devops/stages/9-promotion.md)

![#008000](https://placehold.co/15x15/008000/008000.png) &nbsp; **Video**

[![Glossary app PV](https://img.youtube.com/vi/LBqgitY_j5A/0.jpg)](https://youtu.be/LBqgitY_j5A "Glossary app PV")

---

### 4.2. E-Commerce Service

📰 Currently in development

- [1-Plan (Draft)](services/e_commerce_service/devops/stages/1-plan.md)
  - [Deployment diagram](resources_readme/services/e_commerce_service/diagrams/deployment.svg?raw=1)
- 3-Verify
  - coverages
    - [backend](https://wbfw109.github.io/services/e_commerce_service/coverages/backend/index.html)

#### 4.2.1. Pipeline: GitHub Actions

- [Backend test coverage CD to GitHub Pages](.github/workflows/e_commerce_service-coverage-github_pages-CD.yml)

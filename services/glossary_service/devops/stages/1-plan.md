# Plan

---

Updated on 📅 2023-01-28 23:43:24

- [Plan](#plan)
  - [1. Plan](#1-plan)
    - [1.1. Motivation](#11-motivation)
    - [1.2. Requirements](#12-requirements)
      - [1.2.1. Functional Requirement](#121-functional-requirement)
      - [1.2.2. Non-Functional Requirement](#122-non-functional-requirement)
  - [2. Define](#2-define)
    - [2.1. Nodes](#21-nodes)

## 1. Plan

### 1.1. Motivation

I want to learn glossaries in environment where the words I described are shown as searchable, collapsible tree data structure.

&nbsp;

---

### 1.2. Requirements

#### 1.2.1. Functional Requirement

- Data is provided as static page from file "ref/computer_science_words_korean.json".
  - static page is updated whenever only API that instructs to update static page is performed.
- Users can this App in 2 platform: (Slack, Web)
  1. Slack bot provides Q&A system.
      1. Question: a user queries a word
      2. Answer: Slack bot responses on the word.
          - If one of available list matches, provide that description.
          - If not matches proposes some clickable words. and if user clicks that, provide that description.
  2. Web provides the pages with 2 main components: (Left sidebar, Body section), and side component.
      - Common
        - All element have \<is_folded\> state by a main component.
        - Default value of \<is_folded\> is "True" so that it shows elements in minimum node level (level 1) in Left sidebar component.
      - Common function
        - (Fold | Unfold) all elements: apply to both Left Sidebar and Body section
        - When a user clicks the element in a context of main component  
          , the element \<is_folded\> is toggled.
        - When a user trigger event: (mouse down longer than for 0.5 ~ 1 second and mouse up) to the element in a context of main component  
          , \<is_folded\> of all sub-elements of the element is toggled based on \<is_folded\> state of first child of the element.
        - When a user **~~_double-click_~~** the element in a context of main component  
          , move screen of user browser to anchor of the element in Body section.  
           and (highlight the element for a few seconds, update current breadcrumb).  
          ➡️ Event on click and double-click can not be separate. so I change the condition: when: "click a icon."
      - Main Components
        1. Left Sidebar
            - Name of Elements additionally have prefix as a icon that shows whether \<is_folded\> state is "True" or not.
        2. Body section
            - Elements additionally have description, and Name of Elements have prefix formed by formula: ( (node_level * 2 space) + emoji according to node level ).
            - Breadcrumb that indicates current node hierarchy is always shown as on the top of screen of user browser.
      - Side component: Search bar  
        When a user search characters, words including that characters are filtered and provided as vertical list of Breadcrumb.  
        and if user clicks a word, move the screen of user browser to that element in Body section as \<is_folded\> state of elements of the breadcrumb is set as true.

#### 1.2.2. Non-Functional Requirement

📰 To be determined

&nbsp;

---

## 2. Define

### 2.1. Nodes

1. **Pynecone** (Full stack Framework) ; 🔓 [_Apache License_](https://github.com/pynecone-io/pynecone/blob/main/LICENSE)  
    Set of (**FastAPI** ; 🔓 [_MIT license_](https://github.com/tiangolo/fastapi/blob/master/LICENSE)
    , **NextJS** ; 🔓 [_MIT license_](https://github.com/vercel/next.js/blob/canary/license.md)
    , **React** ; 🔓 [_MIT license_](https://github.com/facebook/react/blob/main/LICENSE) and **Chakra-ui** ; 🔓 [_MIT license_](https://github.com/chakra-ui/chakra-ui/blob/main/LICENSE)
    , **bun** ; 🔓 [_MIT license_](https://github.com/oven-sh/bun#license))
    - Font: Noto fonts ; 🔓 [_SIL Open Font License_](https://github.com/notofonts/noto-fonts/blob/master/LICENSE)
2. Version Control: **GitHub**

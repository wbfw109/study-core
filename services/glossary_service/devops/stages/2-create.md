# Create

---

Updated on 📅 2023-01-29 07:34:17

- [Create](#create)
  - [1. Work breakdown structure](#1-work-breakdown-structure)
  - [2. configuration](#2-configuration)

## 1. Work breakdown structure

with **Agile** software development practices

1. Pynecone (Web fullstack framework)
    - ✅ init and add tasks in task.json that run Pynecone server in Development or Production mode.
    - ✅ Load data.json (nested form) and wrap with pc.Base
      - ✅ make Serializable class (not nested form)
      - ✅ make dictionary to (children, descendants, ascendants) indexes from a index  
        , related with AccordionItem Component through one-time explore data.json.
    - ✅ Divide screen into two main component: (Side bar, Body section)
      - ✅ construct sticky Top menu bar component.
      - ✅ Side bar
        - ✅ construct Accordion component (nested form)  
          Side bar have only word name by node level.
        - ❓ make link component that move to AccordionItem component of body section.  
          ➡️ refer to Feature Request in [3-verify](./3-verify.md):  
          　\[Feature Request\] Can I delay Link to href until on_click event render is complete?
        - ✅ add Side bar into Drawer component from menu icon button of Top menu bar component.
      - ✅ Body section
        - ✅ construct Accordion component (nested form)  
          Body section have word name with description by node level.
        - ✅ add id props and make permalink button icon for each AccordionItem.
        - ✅ add breadcrumb as MenuItem components for each AccordionItem.
      - ✅ (Side bar, Body section)
        - ✅ construct sticky AccordionItem on only 1 level depth.
        - ✅ add button that toggles all accordion item of words.
        - ✅ add on-click event that toggles a accordion item of a word.
        - ✅ add on-1-second-mouse-down event that toggles all sub-accordion items of a word.
        - ✅ add on-click event that move screen from element of Side bar to element of Body section (as permalink).
      - ✅ add tooltips
    - add search bar using overlay  
      ➡️ More study required (; implementation not using wrapped react component).
    - build app in production-mode and public hosting.  
      ➡️ Pynecone currently not support public hosting.
2. Slack bot
  ➡️ Pynecone currently not support public hosting. (plan to use [Backend API Routes](https://pynecone.io/docs/advanced-guide/apiroutes))

## 2. configuration

📰 To be determined

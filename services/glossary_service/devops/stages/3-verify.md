# Verify

---

Updated on 📅 2023-01-29 07:34:31

- [Verify](#verify)
  - [1. Testing](#1-testing)
    - [1.1 Minor point](#11-minor-point)
  - [2. Intermediate assessment](#2-intermediate-assessment)
    - [2.1. Limitation](#21-limitation)
    - [2.2 Issue tracking](#22-issue-tracking)
    - [2.3. Feature request or Discussion](#23-feature-request-or-discussion)

## 1. Testing

### 1.1 Minor point

- When click the Menu item of AccordionItem, remains the tooltip: "BreadCrumb". not disappeared.
- Only on header in 1 depth level, to move screen to permalink not works due to sticky header in 1 depth level

## 2. Intermediate assessment

### 2.1. Limitation

Currently version of **pynecone** is [Public Alpha](https://github.com/pynecone-io/pynecone).

- [Pynecone can not be distributed for external access.](https://pynecone.io/docs/hosting/deploy)
- Pynecone can not control based on event target.
- Pynecone Event handler
  - can only have 0 or 1 arguments, otherwise it occurs ValueError like
    ```ValueError: Lambda <function ...> must have 0 or 1 arguments.```
  - must be in pc.State, otherwise it occurs error ```pydantic.error_wrappers.ValidationError: 1 validation error for EventChain
events -> 0  none is not an allowed value (type=type_error.none.not_allowed)```
  - can not use keyword like ```lambda x: State.a_event_handler(index=current_index)```
- Pynecone can not interpret Self referenced nested type in pc.State. ; Error: maximum recursion
- Even if a data class is serializable, you must new data class that inherits "pc.Base".
    (probably guessing to be caused by Pydantic)
    And you must avoid using some variables, that are reserved word used on BaseVar in this framework.
        E.g. "name"
- When use text component, it can not render multiple whitespace even if "\&nbsp;"  
  Ad-hoc: use "　" specially converted character from "ㄱ".
- Pynecone does not supports focus (ref).

### 2.2 Issue tracking

- ✅ [Build error](https://github.com/pynecone-io/pynecone/issues/292)
- ✅ **\<Bug\>** ( Component: AccordionItem) in Nested function (loop) ): argument value is fixed when register event until return.  
  The bug seems to be fixed in Pynecone version 0.1.13.  
- ➡️ [Argument type is converted in Event chain](https://github.com/pynecone-io/pynecone/issues/342)
- ➡️ [Messages that delay server](https://github.com/pynecone-io/pynecone/issues/360)
- ➡️ [Mouse click event is triggered more slowly on AccordionItem in DrawerBody.](https://github.com/pynecone-io/pynecone/issues/364)
- ➡️ [Event chains can not allow two event handlers at the same time: (handlers with and without argument)](https://github.com/pynecone-io/pynecone/issues/373)

### 2.3. Feature request or Discussion

- ➡️ [\[Discussion\] manipulate based on event target](https://github.com/pynecone-io/pynecone/discussions/297)
- ➡️ [\[Feature Request\] Can I delay Link to href until on_click event render is complete?](https://github.com/pynecone-io/pynecone/issues/374)

# Verify

---

Updated on 📅 2023-02-09 11:38:25

- [Verify](#verify)
  - [1. Testing](#1-testing)
    - [1.1. Limitation and issue tracking](#11-limitation-and-issue-tracking)

## 1. Testing

### 1.1. Limitation and issue tracking

- VSCode tasks with dependsOn does not cancle next task even if a error occurs in current running task.
  - ➡️ [errors in dependsOn background tasks do not prevent subsequent tasks from executing](https://github.com/microsoft/vscode/issues/70283)
    - 🚡 Temp solution: to invoke shell script in a task.

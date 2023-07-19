# WSL2 Environment settings

It is the Guide of Development environment to prevent from some errors.

- [WSL2 Environment settings](#wsl2-environment-settings)
  - [Done](#done)
    - [/etc/resolv.conf](#etcresolvconf)
  - [TO DO (🔍)](#to-do-)

## Done

### /etc/resolv.conf

- ```txt
  nameserver 8.8.8.8
  nameserver 8.8.4.4
  nameserver 1.1.1.1
  nameserver <your WSL2 IPv4 Address>
  ```

  - Google public DNS are "8.8.8.8" and "8.8.4.4".

- otherwise, may cause
  - when `apt install` some packages,
    - get Error `Failed to fetch ... Connection failed ... E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?`.
  - when use other commands, the process may be stuck such as in poetry.

## TO DO (🔍)

- in "mcr.microsoft.com/devcontainers/cpp:ubuntu-18.04" container environment,

  - ```txt
    "systemd" is not running in this container due to its overhead.
    Use the "service" command to start services instead. e.g.:
    ```

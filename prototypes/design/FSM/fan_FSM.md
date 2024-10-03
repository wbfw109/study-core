<!--
# B1: Button 1 (On | Off)
# B2: Button 2 (Speed control in either (Auto mode | Manual mode))
# B2: Button 3 ((Auto mode | Manual mode) toggle; entry point)
# B3: Button 4 (Shutdown Timer setting: None, 3 minute, 5 minute, 7 minute)

-->

```mermaid

stateDiagram-v2
  [*] --> Off
  state "On" as On {
    [*] --> On_Manual

    state "Manual" as On_Manual {
      state "Speed" as On_Manual_Speed {
        [*] --> Low

        Low: Low
        Medium: Medium
        High: High

        Low --> Medium: B2 | USART
        Medium --> High: B2 | USART
        High --> Low: B2 | USART

        Low --> Low
        Medium --> Medium
        High --> High
      }

    }
    
    state "Auto" as On_Auto {
      [*] --> Type1
      Type1: Type1
      Type2: Type2

      Type1 --> Type2: B2 | USART
      Type2 --> Type1: B2 | USART

      Type1 --> Type1
      Type2 --> Type2
    }

    On_Manual --> On_Auto: B3 | UART
    On_Auto --> On_Manual: B3 | UART
  }
  On --> Off: B1 | B4 | USART
  Off --> On: B1 | USART
```
#include <stdio.h>
#include <stdlib.h>

// Function to handle errors.
void error_handling(char *message) {
  // Outputs error message to standard error.
  fputs(message, stderr);
  // Adds newline character to error output.
  fputc('\n', stderr);
  // Terminates program after error.
  exit(1);
}
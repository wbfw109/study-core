// arpa: Advanced Research Projects Agency. Provides definitions for internet
// operations.
#include <arpa/inet.h>
// stdio: Standard Input Output, for basic I/O operations like printf.
#include <stdio.h>
// stdlib: Standard Library, for general functions like memory allocation and
// process control.
#include <stdlib.h>
// string: Provides functions to manipulate C strings (arrays of characters).
#include <string.h>
// sys: System, socket: Socket definitions and protocols for inter-process
// communication.
#include <sys/socket.h>
// unistd: UNIX standard, provides access to the POSIX operating system API.
#include "error_handling.c"
#include <unistd.h>

// Function declaration for error handling.
void error_handling(char *message);

// main function takes command-line arguments.
int main(int argc, char *argv[]) {
  // sock: Integer to store the socket descriptor.
  int sock;
  // sockaddr_in: Structure for storing serverer IP address and port number.
  struct sockaddr_in server_addr;
  // Array to store the message received from serverer.
  char message[30];
  // Integer to store the length of the received message.
  int str_len;

  // argc: Argument count, checks if exactly 3 arguments are passed.
  if (argc != 3) {
    printf("Usage : %s <IP> <port>\n", argv[0]);
    // Exits program if argument count is incorrect.
    exit(1);
  }

  // PF_INET: Protocol Family for Internet, SOCK_STREAM: TCP socket.
  sock = socket(PF_INET, SOCK_STREAM, 0);
  if (sock == -1)
    // Checks for socket creation error.
    error_handling("socket() error");

  // memset: Memory set, initializes server_addr to zero.
  memset(&server_addr, 0, sizeof(server_addr));
  // AF_INET: Address Family for Internet.
  server_addr.sin_family = AF_INET;
  // inet_addr: Converts IP address from text to binary form.
  server_addr.sin_addr.s_addr = inet_addr(argv[1]);
  // htons: Host to Network Short, converts port to network byte order.
  server_addr.sin_port = htons(atoi(argv[2]));

  // connect: Establishes connection with the serverer.
  if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
    error_handling("connect() error!");

  // read: Reads data from socket into message.
  str_len = read(sock, message, sizeof(message) - 1);
  if (str_len == -1)
    // Checks for read error.
    error_handling("read() error!");

  // Displays received message.
  printf("Message from serverer: %s \n", message);
  // close: Closes the socket descriptor.
  close(sock);
  return 0;
}

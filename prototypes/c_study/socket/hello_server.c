
// arpa: Advanced Research Projects Agency. Provides definitions for internet//
// operations.
#include <arpa/inet.h>
// stdio: Standard Input Output, for basic I/O operations like printf.
#include <stdio.h>
// stdlib: Standard Library, for general functions like memory allocation and
// process control.
#include <stdlib.h>
#include <string.h>
// sys: System, socket: Socket definitions and protocols for inter-process
// communication.
#include <sys/socket.h>
// unistd: UNIX standard, provides access to the POSIX operating system API.
#include "error_handling.c"
#include <unistd.h>

// main function takes command-line arguments.
int main(int argc, char *argv[]) {
  // server_sock: Socket descriptor for serverer.
  int server_sock;
  // client_sock: Socket descriptor for client.
  int client_sock;

  // sockaddr_in: Structure for storing serverer address information.
  struct sockaddr_in server_addr;
  // sockaddr_in: Structure for storing client address information.
  struct sockaddr_in client_addr;
  // socklen_t: Data type for storing address sizes.
  socklen_t client_addr_size;

  // Message to send to the client.
  char message[] = "Hello World!";

  // argc: Argument count, checks if exactly 2 arguments are passed.
  if (argc != 2) {
    printf("Usage : %s <port>\n", argv[0]);
    // Exits program if argument count is incorrect.
    exit(1);
  }

  // PF_INET: Protocol Family for Internet, SOCK_STREAM: TCP socket.
  server_sock = socket(PF_INET, SOCK_STREAM, 0);
  if (server_sock == -1) {
    // Checks for socket creation error.
    error_handling("socket() error");
  }

  // memset: Initializes server_addr structure to zero.
  memset(&server_addr, 0, sizeof(server_addr));

  // Set address family for the socket to IPv4 (AF_INET).
  // The sin_family field specifies the address family for the socket,
  // allowing the socket to understand the format and nature of the address.
  // AF_INET indicates that the socket will use the IPv4 protocol.
  server_addr.sin_family = AF_INET;

  // Set the IP address for the socket to INADDR_ANY, which allows the server to
  // accept connections on any available network interface. For example, this
  // could mean any IP address assigned to the machine, like "127.0.0.1"
  // (loopback) or a public IP.
  //
  // The htonl function ("Host to Network Long") is used to ensure that the
  // value is stored in network byte order (big-endian). Network protocols
  // specify big-endian format, while some systems use little-endian by default.
  // The conversion is necessary to make sure all systems interpret the IP
  // address consistently.
  server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  // Convert the port number from the command-line argument into an integer
  // using atoi. The atoi function ("ASCII to Integer") converts a string
  // (argv[1]) to an integer. Since command-line arguments are provided as
  // strings, this step ensures that the provided port number can be used as an
  // actual integer.
  //
  // The htons function ("Host to Network Short") converts this integer from
  // host byte order (which could be little-endian on some machines) to network
  // byte order (big-endian). Like IP addresses, network protocols specify that
  // port numbers should be in big-endian format for consistency across
  // different systems.
  server_addr.sin_port = htons(atoi(argv[1]));

  // bind: Associates the socket with a specific IP and port.
  if (bind(server_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) ==
      -1) {
    error_handling("bind() error");
  }

  // listen: Prepares the socket to accept incoming connections.
  if (listen(server_sock, 5) == -1) {
    error_handling("listen() error");
  }

  // Sets size of client address.
  client_addr_size = sizeof(client_addr);
  // accept: Accepts a client connection.
  client_sock =
      accept(server_sock, (struct sockaddr *)&client_addr, &client_addr_size);
  if (client_sock == -1) {
    // Checks for accept error.
    error_handling("accept() error");
  }

  // write: Sends message to the client.
  write(client_sock, message, sizeof(message));
  // close: Closes client socket.
  close(client_sock);
  // close: Closes serverer socket.
  close(server_sock);
  return 0;
}

/*
❓ About struct sockaddr_in
  Reference
    ⚓ sockaddr ; https://man7.org/linux/man-pages/man3/sockaddr.3type.html
      Internet domain sockets

  It will use the created the variable name "sin_family" by implementation
  ```cpp
    #define	__SOCKADDR_COMMON(sa_prefix) \
    sa_family_t sa_prefix##family
    struct sockaddr_in
    {
      __SOCKADDR_COMMON (sin_);
      ...
    }
  ```
  ❔ The `__SOCKADDR_COMMON` Macro:
    The `__SOCKADDR_COMMON` macro defines fields that are common to all `sockaddr` structures.
    It is used to declare the `sa_family` field with the prefix `sin_`, which becomes `sin_family` when combined.
    The `##` in `sa_prefix##family` is a preprocessor concatenation operator, allowing the macro to prepend `sin_` to `family`, forming `sin_family`.


  Explanation:
    The `struct sockaddr_in` structure is specifically designed for handling IPv4 internet addresses in network programming.
    This structure includes key fields that specify the address family, port number, and IP address.

    Key Fields in `sockaddr_in`:
    - `sin_family`: Specifies the address family. In this case, `AF_INET` is used to indicate IPv4.
    - `sin_port`: Stores the port number for the socket. This value is stored in "network byte order" (big-endian format), 
      as required by network protocols. The `htons` (Host to Network Short) function is used to convert the port 
      number from the host’s byte order to network byte order.
    - `sin_addr`: Stores the IP address for the socket, specifically in `sin_addr.s_addr`. The function `htonl` (Host 
      to Network Long) is used to convert the address to network byte order.

  Additional Details:
    - `AF_INET`: This constant represents the address family for IPv4. It is used to define the `sin_family` field to 
      ensure the socket knows it will be communicating over IPv4.
    - `INADDR_ANY`: This constant is used with `sin_addr.s_addr` to allow the socket to accept connections on any 
      network interface associated with the host (e.g., any available IP address). It is converted to network 
      byte order using `htonl`.
    - `htons` and `htonl`: Both functions are used to convert values from host byte order (which may be little-endian 
      or big-endian depending on the machine) to network byte order (always big-endian). This ensures consistency 
      across different networked systems.
    - `atoi`: Converts a string to an integer. In this context, it is used to interpret the port number provided as a 
      command-line argument and convert it from a string to an integer that can be used in `sin_port`.



  TODO: when describe, dexplan why I'm using ... each AF_INET, htonl,
INADDR_ANY, htons, atoi,.. and what are these.

ntohs
https://serblin.tistory.com/5
*/

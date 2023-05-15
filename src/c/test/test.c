#include <stdio.h>
#define _countof(array) (sizeof(array) / sizeof(array[0]))

#include <avr/io.h>
#include <util/delay.h>

#define LED_DELAY 100 // LED가 켜지는 간격 (ms)

int main(void)
{
  DDRD = 0xFF;                  // 포트 D를 출력으로 설정 (8개의 LED를 제어하기 위함)
  uint8_t pattern = 0b00000001; // 초기 패턴 (첫 번째 LED 켜짐)
  int8_t direction = 0;         // LED 진행 방향 (1: 왼쪽으로, 0: 오른쪽으로)

  while (1)
  {
    PORTD = pattern;      // 현재 패턴을 포트 D에 출력
    _delay_ms(LED_DELAY); // LED가 켜진 상태로 유지

    PORTD = 0x00;         // 모든 LED 끄기
    _delay_ms(LED_DELAY); // LED가 꺼진 상태로 유지

    // reverse direction
    if (pattern == 0b10000000 || pattern == 0b00000001)
    {
      direction ^= 1;
    }

    direction ?: (pattern <<= 1) : (pattern >>= 1);
  }
}

float centi_to_meter(void)
{
  printf("===== Centi to Meter =====\n");
  printf("input num (unit: cm) (limit: 1 ~ 200): ");

  int num;
  while (1)
  {
    scanf("%d", &num);
    if (num < 1 || num > 200)
    {
      printf("Please enter the range 1 ~ 200\n");
    }
    else
      break;
  }

  return num / 100.0;
}

// parameter sums: {sum_even_i, sum_odd_i}
void *get_sum_pair(int *sums)
{
  int num[10];
  int sum_even_i = 0;
  int sum_odd_i = 0;
  size_t i;

  for (i = 0; i < _countof(num); i++)
  {
    // input
    num[i] = i + 1;

    // process
    (i % 2 == 0) ? (sum_even_i += num[i]) : (sum_odd_i += num[i]);
  }

  sums[0] = sum_even_i;
  sums[1] = sum_odd_i;
}

void fizz_buzz(int num)
{
  printf("===== FizzBuzz =====\n");
  size_t i;
  char is_fizz_or_buzz = 0;

  for (i = 1; i <= num; i++)
  {
    is_fizz_or_buzz = 0;
    if (i % 3 == 0)
    {
      is_fizz_or_buzz = 1;
      printf("Fizz");
    }
    if (i % 5 == 0)
    {
      is_fizz_or_buzz = 1;
      printf("Buzz");
    }

    if (is_fizz_or_buzz == 0)
    {
      printf("%zd", i);
    }

    printf("\n");
  }
}

int print_grade_average(void)
{
  /*
  Code review
    - 메모리 사용량을 줄이자.
      >>> 변수의 데이터 형식 변경.

  */
  unsigned char kor = 3;
  unsigned char eng = 5;
  unsigned char mat = 4;

  unsigned int credits = kor + eng + mat;

  float kscore = 3.8;
  float escore = 4.4;
  float matscore = 3.9;
  float grade = (kor * kscore + eng * escore + mat * matscore) / credits; // Implicit Type casting.

  char res = (credits >= 10) && (grade >= 4.0); // Short-circuiting.
  // instead can use If condition statement, switch-case statement or Ternary operator.

  printf("%d\n", res);

  return 0;
}

// int main(void)
// {
//   char s[10] = "abcdefghi";
//   printf("%ld", _countof(s));
// }

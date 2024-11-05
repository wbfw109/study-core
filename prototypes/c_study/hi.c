#include <stdio.h>

int input();
int output(int num);
int main(void){
    int n = input();
    if(!(output(n))) {
        printf("Good bye!\n");
    }
    return 0;
}

// gcc hello.c -o hello

int input(){
    int num;
    printf("enter a number: ");
    if (scanf("%d", &num) < 1) {
        printf("Invalid input!\n");
        return -1;
    }
    return num;
}

int output(int num){
    int i;
    for (i=0; i<num; i++){
        printf("Hi\n");
    }
    return 0;
}


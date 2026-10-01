#include <stdio.h>

int main() {
int n;
scanf("%d", &n);
int T[n] ;
for (int i = 0; i < n; i++){
    T[i]=0;
}
int x;
for (int i = 0; i < n; i++) {
    scanf("%d", &x);
    T[x]++;
}
int cnt=T[1];
int m=0;
for (int i = 0; i < n; i++){
    if(cnt<T[i]){
        cnt=T[i];
        m=i;
    }
}
printf("%d\n",cnt);
printf("%d\n",m);







return 0;
}   
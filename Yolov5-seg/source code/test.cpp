#include <iostream>

int main(){
    int arr[]={1,2,3,4};
    const int* str = &arr;
    std::cout<<&str+0<<std::endl;
    std::cout<<&str+1<<std::endl;
    std::cout<<&str+2<<std::endl;
    std::cout<<&str+3<<std::endl;
}
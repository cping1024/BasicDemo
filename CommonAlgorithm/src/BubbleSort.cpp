#include <stdio.h>
#include <assert.h>
#include <vector>

void bubble_sort(int *array, int len) {
    assert(array != nullptr);

    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                int temp = array[j + 1];
                array[j + 1] = array[j];
                array[j] = temp; 
            } 
        }
    }
}

int main(int argc, char* argv[]) {
    
    int num = 0;
    std::vector<int> input;
    while(scanf("%d", &num) != EOF) {
        input.push_back(num);
    } 

    printf("input array:\n");
    for (int i = 0; i < input.size(); ++i) {
        printf("%d ", input[i]);
    }
    printf("\n");

    bubble_sort(input.data(), input.size());


    printf("sorted array:\n");
    for (int i = 0; i < input.size(); ++i) {
        printf("%d ", input[i]);
    }
    printf("\n");

    return 0;
}

#include <stdio.h>
#include <vector>

void QuickSort(int* array, int low, int high) {
    if (low >= high) return;

    int first = low;
    int last = high;
    int key = array[low];

    while (first < last) {
        while (first < last && array[last] >= key) {
            --last;
        }

        array[first] = array[last];

        while(first < last && array[first] <= key) {
            ++first;
        }

        array[last] = array[first];
    }

    array[first] = key;
    QuickSort(array, low, first - 1);
    QuickSort(array, first + 1, high);
}

int main(int argc, char* argv[]) {

    int i = 0;  
    std::vector<int> input;
    while (scanf("%d", &i) != EOF) {
        input.push_back(i);
    }

    printf("\ninput array:\n");
    for (int i = 0; i < input.size(); ++i) {
        printf("%d ", input[i]);
    }
    printf("\n");

    QuickSort(input.data(), 0, input.size() - 1);

    printf("sorted array:\n");
    for (int i = 0; i < input.size(); ++i) {
        printf("%d ", input[i]);    
    }
    printf("\n");

    return 0;
}

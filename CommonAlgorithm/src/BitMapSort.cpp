#include <stdio.h>
#include <vector>
#include <string.h>

#define BITSPERWORD     (32)
#define MAX_NUM     (10000000)
#define MASK        (0X1F)
#define SHIFT       (5)

/// bitmap array
int Array[MAX_NUM/8 + 1];

/// @brief find relate bit and set
void setBit(int num)
{
    Array[num >> SHIFT] |= 1 << (num & MASK);
}

/// @brief reset all bit zero
void resetAllBit()
{
    memset(Array, 0, sizeof(Array));
}

/// @brief test input num is mapped
int testBit(int num)
{
    return Array[num >> SHIFT] & 1 << (num & MASK);
}

/// @brief simple bitmap sort implement
/// @param[in] input unsorted num array
/// @param[out] output sorted num array
/// @note time complexity is MAX_NUM, space complexity is MAX_NUM/8 byte
void BitMapSort(const std::vector<int>& input, std::vector<int>& output) {
    if (input.empty()) return;

    resetAllBit();

    size_t len = input.size();
    for (size_t ix = 0; ix < len; ++ix) {
        setBit(input[ix]);
    }

    /// get output from bitmap
    for (size_t ix = 0; ix < MAX_NUM; ++ix) {
        if (testBit(ix)) {
            output.push_back(ix);
        }
    }
}

int main(int argc, char* argv[]) {
    
    std::vector<int> input;
    int i = 0;
    while(scanf("%d", &i) != EOF) {
        input.push_back(i);
    }
    
    printf("input before sort:\n");
    for (size_t i = 0; i < input.size(); ++i) {
        printf("%d ", input[i]);
    }
    printf("\n");


    std::vector<int> output;
    BitMapSort(input, output);

    printf("output after sort:\n");
    for (size_t i = 0; i < output.size(); ++i) {
        printf("%d ", output[i]);
    }
    printf("\n");

	return 0;
}

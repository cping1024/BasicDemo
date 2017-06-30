#include <stdio.h>
#include <string.h>

int FilterSymbolFromeString(const char* src, char *output) {
	if (!src || !output) return -1;

	int left = 0;
	int right = 0;
	size_t len = strlen(src);
	for (size_t i = 0; i < len; ++i) {
		if (src[i] == '(') ++left;
		else if (src[i] == ')') ++right;
		else strncat(output, &src[i], 1);
	}

	if (left != right) return -1;

	return 0;
}

int main(int argc, char* argv[]) {
	
	const char *src = "(9, (1,5), (2,8),(4,(1,8), 8, ) ,0)";
	char output[128];
	memset(output, 0, sizeof(output));

	printf("src: %s. \n", src);

	/// filter
	int ret = FilterSymbolFromeString(src, output);
	if (ret != 0) printf("valid args.\n");

	printf("filter: %s.\n", output);
	return 0;
}

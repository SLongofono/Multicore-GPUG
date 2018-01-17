#include <stdio.h>
#include <stdlib.h>

void mergesort(int *src, int *scratch, int len);
void partition(int* right, int* left, int begin, int end);
void merge(int *src, int *dest, int begin, int center, int end);
void copyArray(int *src, int*dest, int begin, int end);
void printList(int *ls, int len, int toFile);


int main(void){
	int len = 100;		// number of values
	int lefty[len];		// list of values
	int righty[len];	// scratch array
	for(int i = len; i>0; --i){
		lefty[len-i] = i;
	}

	// Sorted array in lefty, garbage in righty
	mergesort(lefty, righty, len);
	printList(lefty, len, 1);
	return 0;
}


// Prints to stdout if toFile = 0, otherwise prints to sorted.txt in the working
// directory.
void printList(int *ls, int len, int toFile){
	int *curr = ls;
	int *end = &(*(ls+len));

	if(toFile){
		FILE *outfile = fopen("sorted.txt", "w");
		if(0 != outfile){
			while(curr != end){
				fprintf(outfile, "%d ", *curr);
				curr++;
			}
			fprintf(outfile, "\n");
		}
		fclose(outfile);
	}
	else{
		while(curr != end){
			printf("%d ", *curr);
			curr++;
		}
		printf("\n");
	}
}


// Top-down mergesort using a source and work array.  Assumes they are of equal
// length and accessible to this context
void mergesort(int *src, int *scratch, int len){
	// Create duplicate array in right
	copyArray(src, scratch, 0, len);

	partition(scratch, src, 0, len);
}


void partition(int *right, int *left, int begin, int end){
	if (end-begin < 2){
		// list is size 1, nothing to do
		return;
	}
	int center = (begin + end)/2;

	// Sort first half
	partition(left, right, begin, center);

	// Sort second half
	partition(left, right, center, end);

	// merge results
	merge(right, left, begin, center, end);
}


/*
 * Assumes that two lists are stored consecutively in src, ready to be sorted into dest
 */
void merge(int *src, int *dest, int begin, int center, int end){
	int leftIndex = begin;
	int rightIndex = center;
	for(int curr = begin; curr < end; ++curr){
		if(leftIndex < center){
			if(rightIndex >= end || 
			   src[leftIndex] <= src[rightIndex]){
				dest[curr] = src[leftIndex];
				leftIndex++;
			}
			else{
				dest[curr] = src[rightIndex];
				rightIndex++;
			}
		}
	}
}

void copyArray(int *src, int *dest, int begin, int end){
	for(int i = begin; i < end; ++i){
		dest[i] = src[i];
	}
}

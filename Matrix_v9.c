#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>

#define ROWS 10000
#define COLUMNS 10000

int i,j,k;
FILE *fp;

double** matrixA;
double** matrixB;
double** matrixC;
double** A11;
double** A12;
double** A21;
double** A22;

double** B11;
double** B12;
double** B21;
double** B22;

double** C11;
double** C12;
double** C21;
double** C22;

double** M1;
double** M2;
double** M3;
double** M4;
double** M5;
double** M6;
double** M7;

double** addMat1;
double** addMat2;
double** addMat3;
double** addMat4;
double** addMat5;
double** addMat6;

double** subMat1;
double** subMat2;
double** subMat3;
double** subMat4;

int generateRandomNumber()
{
	return rand()%10 + 1;
}

void printMatrix(double** mat, int rows, int cols)
{
	for(i = 0; i < rows; ++i) 
	{
		for (j = 0; j < cols; ++j) 
		{
			fprintf(fp, "%lf ", mat[i][j]);
		}
		fprintf(fp, "\n");
	}
}

void printTime(clock_t start, clock_t end)
{
	fprintf(fp, "Time taken: %lf seconds\n", ((double) (end - start)) / CLOCKS_PER_SEC);
	fflush(fp);
}

void initMatrix(double** a, double** b)
{
	for(i=0; i<ROWS; i++)
	{
	    	for(j=0; j<COLUMNS; j++)
		{
		    	a[i][j] = generateRandomNumber();
			b[i][j] = generateRandomNumber();
	    	}
      	}
}

void memoryAllocate()
{
	matrixA = (double **)malloc(ROWS * sizeof(double *));
	matrixB = (double **)malloc(ROWS * sizeof(double *));
	matrixC = (double **)malloc(ROWS * sizeof(double *));
	for (i=0; i<ROWS; i++)
	{
        	matrixA[i] = (double *)malloc(COLUMNS * sizeof(double));
		matrixB[i] = (double *)malloc(COLUMNS * sizeof(double));
		matrixC[i] = (double *)malloc(COLUMNS * sizeof(double));
	}
	fp = fopen("/tmp/benchmark.txt", "a+");
}

void memoryAllocateStrassen()
{
	A11 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	A12 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	A21 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	A22 = (double **)malloc(ROWS/2 * sizeof(double *)); 

	B11 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	B12 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	B21 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	B22 = (double **)malloc(ROWS/2 * sizeof(double *)); 

	C11 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	C12 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	C21 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	C22 = (double **)malloc(ROWS/2 * sizeof(double *)); 

	M1 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	M2 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	M3 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	M4 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	M5 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	M6 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	M7 = (double **)malloc(ROWS/2 * sizeof(double *)); 

	addMat1 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	addMat2 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	addMat3 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	addMat4 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	addMat5 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	addMat6 = (double **)malloc(ROWS/2 * sizeof(double *)); 

	subMat1 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	subMat2 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	subMat3 = (double **)malloc(ROWS/2 * sizeof(double *)); 
	subMat4 = (double **)malloc(ROWS/2 * sizeof(double *)); 

	for (i=0; i<ROWS/2; i++)
	{
        	A11[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		A12[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		A21[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		A22[i] = (double *)malloc(COLUMNS/2 * sizeof(double));

        	B11[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		B12[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		B21[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		B22[i] = (double *)malloc(COLUMNS/2 * sizeof(double));

        	C11[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		C12[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		C21[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		C22[i] = (double *)malloc(COLUMNS/2 * sizeof(double));

        	M1[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		M2[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		M3[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		M4[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		M5[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		M6[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		M7[i] = (double *)malloc(COLUMNS/2 * sizeof(double));

		addMat1[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		addMat2[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		addMat3[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		addMat4[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		addMat5[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		addMat6[i] = (double *)malloc(COLUMNS/2 * sizeof(double));

		subMat1[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		subMat2[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		subMat3[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
		subMat4[i] = (double *)malloc(COLUMNS/2 * sizeof(double));
	}
}

void multiplyMatrix(double **a, double **b, double **c, int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd)
{
	int ilocal, jlocal, klocal;
    	for(ilocal = iStart; ilocal < iEnd; ++ilocal)
    	{
    		for(jlocal = jStart; jlocal < jEnd; ++jlocal)
    		{
    			c[ilocal][jlocal] = 0;
    			for(klocal = kStart; klocal < kEnd; klocal++)
    			{
    				c[ilocal][jlocal] += a[ilocal][klocal] * b[klocal][jlocal];
			}
		}
	}
}

void singleThreadFunction()
{
	clock_t startTime,endTime;
	float timeDifference;

	fprintf(fp, "\n Normal Multiplication \n");
	fflush(fp);
	initMatrix(matrixA, matrixB);

	startTime = clock();
	multiplyMatrix(matrixA, matrixB, matrixC, 0, ROWS, 0, COLUMNS, 0, ROWS);
    	endTime = clock();
	printTime(startTime, endTime);
//	printMatrix(matrixC, ROWS, COLUMNS);
}


void openMPMultiplication(int numberThreads)
{
	int numberOfThreads;
	clock_t startTime,endTime;
	float timeDifference;
	numberOfThreads = numberThreads;
   	fprintf(fp, "\n OpenMP Multiplication ");
	fflush(fp);
	initMatrix(matrixA, matrixB);
	
      	omp_set_num_threads(numberOfThreads);
	
	//printf("\n Number of threads : %i \n", omp_get_num_threads());
	
	startTime = clock();
	#pragma omp parallel for private(k,j)
	for(i = 0; i < ROWS; i++)
	{
		for(j = 0; j < COLUMNS;j++)
		{
    			matrixC[i][j] = 0;
    			for(k = 0; k < ROWS; k++)
    			{
    				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
				
			}
		}
	}
	endTime = clock();
	printTime(startTime, endTime);
//	printMatrix(matrixC, ROWS, COLUMNS);
}

void multiThreadMultiplication(int numberThreads)
{		
	clock_t startTime,endTime;
	float timeDifference;
   	fprintf(fp, "\n MultiThread Multiplication ");
	fflush(fp);
	initMatrix(matrixA, matrixB);

	int numberOfThreads = numberThreads;
	fprintf(fp, "\n Number of threads passed in the command line argument is %d \n",numberOfThreads);
	pthread_t *thread;
        thread = (pthread_t*) malloc (numberOfThreads * sizeof(pthread_t));

	//thread function takes slice as its argument
	void *multiply(void *slice)
	{
		int s = (intptr_t) slice;
		int from = (s*ROWS)/numberOfThreads;
		int to = ((s+1)*COLUMNS)/numberOfThreads;
//		printf("\n slice Number : %d (from rows %d to %d) \n",s,from,to-1);
		multiplyMatrix(matrixA, matrixB, matrixC, from, to, 0, COLUMNS, 0, ROWS);
	}

	startTime = clock();
	for(i=0; i<numberOfThreads; i++)
	{
		if(pthread_create(&thread[i], NULL, multiply, (void *)(intptr_t)i) != 0)
		{
			fprintf(fp, "\n Can not create thread ");
			free(thread);
			exit(-1);
		}
	}

	for(i = 0; i < numberOfThreads; i++)
	{
		pthread_join(thread[i],NULL);
	}
	endTime = clock();
	printTime(startTime, endTime);
//	printMatrix(matrixC, ROWS, COLUMNS);
}

void loopUnrollingMultiplication()
{		
	clock_t startTime,endTime;
	float timeDifference;
	double temp;
   	fprintf(fp, "\nLoop Unrolling Multiplication\n");
	fflush(fp); 
	initMatrix(matrixA, matrixB);

	startTime = clock();
	for(i=0;i<ROWS;i++)
	{
		for(j=0;j<COLUMNS;j++)
		{
			temp=0;
			for(k=0;k < ROWS-7; k+=8)
			{
				temp += matrixA[i][k] * matrixB[k][j];
				temp += matrixA[i][k+1] * matrixB[k+1][j];
				temp += matrixA[i][k+2] * matrixB[k+2][j];
				temp += matrixA[i][k+3] * matrixB[k+3][j];
				temp += matrixA[i][k+4] * matrixB[k+4][j];
				temp += matrixA[i][k+5] * matrixB[k+5][j];
				temp += matrixA[i][k+6] * matrixB[k+6][j];
				temp += matrixA[i][k+7] * matrixB[k+7][j];
			}
			for(;k<ROWS;k++)
			{
				temp += matrixA[i][k] * matrixB[k][j];
			}
			matrixC[i][j]=temp;
		}
	}
	endTime = clock();
	printTime(startTime, endTime);
//	printMatrix(matrixC, ROWS, COLUMNS);
}

void strassenMultiplication(int numberThreads)
{
	clock_t startTime,endTime;
	float timeDifference;
	fprintf(fp, "\nStrassen Algorithm Multiplication \n");
	fflush(fp);
	initMatrix(matrixA, matrixB);

	startTime = clock();
	int splitR = ROWS/2;
	int splitC = COLUMNS/2;
	int numberOfThreads = numberThreads;
	pthread_t *thread;
        thread = (pthread_t*) malloc (numberOfThreads * sizeof(pthread_t));

	for(i=0;i<splitR;i++)
	{
		for(j=0;j<splitC;j++)
		{
			A11[i][j] = matrixA[i][j];
			B11[i][j] = matrixB[i][j];


			A12[i][j] = matrixA[i][j+splitC];
			B12[i][j] = matrixB[i][j+splitC];

			A21[i][j] = matrixA[i+splitR][j];
			B21[i][j] = matrixB[i+splitR][j];

			A22[i][j] = matrixA[i+splitR][j+splitC];
			B22[i][j] = matrixB[i+splitR][j+splitC];
		}
	}

	for(i=0;i<splitR;i++)
	{
		for(j=0;j<splitC;j++)
		{
			addMat1[i][j] = A11[i][j] + A22[i][j];
			addMat2[i][j] = B11[i][j] + B22[i][j];
			addMat3[i][j] = A21[i][j] + A22[i][j];
			addMat4[i][j] = A11[i][j] + A12[i][j];
			addMat5[i][j] = B11[i][j] + B12[i][j];
			addMat6[i][j] = B21[i][j] + B22[i][j];

			subMat1[i][j] = B12[i][j] - B22[i][j];
			subMat2[i][j] = B21[i][j] - B11[i][j];
			subMat3[i][j] = A12[i][j] - A22[i][j];
			subMat4[i][j] = A21[i][j] - A11[i][j];

			
		}
	}

	//M1 = (A11 + A22) * (B11 + B22)
	//M2 = (A21 + A22) * B11
	//M3 = A11 * (B12 - B22)
	//M4 = A22 * (B21 - B11)
	//M5 = (A11 + A12) * B22
	//M6 = (A21 - A11) * (B11 + B12)
	//M7 = (A12 - A22) * (B21 + B22)

	for(i=0; i<splitR; i++)
	{
		for(j=0; j<splitC; j++)
		{
			M1[i][j] = 0.0; M2[i][j] = 0.0; M3[i][j] = 0.0; M4[i][j] = 0.0; M5[i][j] = 0.0; M6[i][j] = 0.0; M7[i][j] = 0.0;
			for(k=0; k<splitR; k++)
			{
				M1[i][j] += addMat1[i][k] * addMat2[k][j];
				M2[i][j] += addMat3[i][k]* B11[k][j];
				M3[i][j] += A11[i][k] * subMat1[k][j];
				M4[i][j] += A22[i][k] * subMat2[k][j];
				M5[i][j] += addMat4[i][k]* B22[k][j];
				M6[i][j] += subMat4[i][k]* addMat5[k][j];
				M7[i][j] += subMat3[i][k]* addMat6[k][j];
			}
		}
	}

	//C11 = M1 + M4 -M5 + M7
	//C12 = M3 + M5
	//C21 = M2 + M4
	//C22 = M1 - M2 + M3 + M6

	void *multiplyC11(void *slice)
	{
		int ilocal, jlocal;
		for(ilocal=0;ilocal<splitR;ilocal++)
		{
			for(jlocal=0;jlocal<COLUMNS/2;jlocal++)
			{
				C11[ilocal][jlocal] = M1[ilocal][jlocal] + M4[ilocal][jlocal] - M5[ilocal][jlocal] + M7[ilocal][jlocal];
			}
		}
	}

	void *multiplyC12(void *slice)
	{
		int ilocal, jlocal;
		for(ilocal=0;ilocal<splitR;ilocal++)
		{
			for(jlocal=0;jlocal<COLUMNS/2;jlocal++)
			{
				C12[ilocal][jlocal] = M3[ilocal][jlocal] + M5[ilocal][jlocal];
			}
		}
	}

	void *multiplyC21(void *slice)
	{
		int ilocal, jlocal;
		for(ilocal=0;ilocal<splitR;ilocal++)
		{
			for(jlocal=0;jlocal<splitC;jlocal++)
			{
				C21[ilocal][jlocal] = M2[ilocal][jlocal] + M4[ilocal][jlocal];
			}
		}
	}

	void *multiplyC22(void *slice)
	{
		int ilocal, jlocal;
		for(ilocal=0;ilocal<splitR;ilocal++)
		{
			for(jlocal=0;jlocal<COLUMNS/2;jlocal++)
			{
				C22[ilocal][jlocal] = M1[ilocal][jlocal] - M2[ilocal][jlocal] + M3[ilocal][jlocal] + M6[ilocal][jlocal];
			}
		}
	}

	for(i=0;i<4;i++)
	{
		if(pthread_create(&thread[i++],NULL,multiplyC11,(void *)(intptr_t)i) != 0)
		{
			fprintf(fp, "\n can not create thread ");
			free(thread);
			exit(-1);
		}
		if(pthread_create(&thread[i++],NULL,multiplyC12,(void *)(intptr_t)i) != 0)
		{
			fprintf(fp, "\n can not create thread ");
			free(thread);
			exit(-1);
		}
		if(pthread_create(&thread[i++],NULL,multiplyC21,(void *)(intptr_t)i) != 0)
		{
			fprintf(fp, "\n can not create thread ");
			free(thread);
			exit(-1);
		}
		if(pthread_create(&thread[i++],NULL,multiplyC22,(void *)(intptr_t)i) != 0)
		{
			fprintf(fp, "\n can not create thread ");
			free(thread);
			exit(-1);
		}
	}

	for(i=0;i<4;i++)
	{
		pthread_join(thread[i],NULL);
	}


	for(i=0;i<splitR;i++)
	{
		for(j=0;j<splitC;j++)
		{
			matrixC[i][j] = C11[i][j];
		}
	}

	for(i=0;i<splitR;i++)
	{
		for(j=splitC;j<COLUMNS;j++)
		{
			matrixC[i][j] = C12[i][j-splitC];
		}
	}

	for(i=splitR;i<ROWS;i++)
	{
		for(j=0;j<splitC;j++)
		{
			matrixC[i][j] = C12[i-splitR][j];
		}
	}

	for(i=splitR;i<ROWS;i++)
	{
		for(j=splitC;j<COLUMNS;j++)
		{
			matrixC[i][j] = C12[i-splitR][j-splitC];
		}
	}

	endTime = clock();
	printTime(startTime, endTime);
//	printMatrix(matrixC, ROWS, COLUMNS);
}

int main(int argc, char *argv[])
{
	int numberThreads;
	if(argc != 2)
	{
		fprintf(fp, " \n Enter number of threads to be used in command line argument ");
		exit(1);
	}

	numberThreads=atoi(argv[1]);
	if(numberThreads<1)
	{
		fprintf(fp, " \n Number of threads specified should be more than 1 ");
		exit(1);
	}
	
	memoryAllocate();

	loopUnrollingMultiplication();
	openMPMultiplication(numberThreads);
	singleThreadFunction();
	multiThreadMultiplication(numberThreads);

	memoryAllocateStrassen();
	strassenMultiplication(numberThreads);

	fprintf(fp, "**********************END OF RUN************************\n");
	fclose(fp);
	return 0;
}



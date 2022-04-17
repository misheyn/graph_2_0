#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void getInfo() {
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device name: %s\n", deviceProp.name);
    printf("Total global memory: %ull\n", deviceProp.totalGlobalMem);
    printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
    printf("Registers per block: %d\n", deviceProp.regsPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Memory pitch: %d\n", deviceProp.memPitch);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("Max grid size: x = %d, y = %d, z = %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("Clock rate: %d\n", deviceProp.clockRate);
    printf("Total constant memory: %d\n", deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %d\n", deviceProp.textureAlignment);
    printf("Device overlap: %d\n", deviceProp.deviceOverlap);
    printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n", deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
}

__global__ void subRow(double *matrix, double factor, int firstInd, int secondInd, int size) {
    unsigned int i = threadIdx.x;
    if (i < size)
        matrix[firstInd + i] -= matrix[secondInd + i] * factor;
}

__global__ void abobaFunc(double *matrix, double first, int order, int scale, unsigned blockX, unsigned threadX) {
    unsigned i = (order + 1) * (order - scale) + order * (threadIdx.x + 1);
    double mult = matrix[i] / first;
    subRow<<<blockX, threadX>>>(matrix, mult, i, (order + 1) * (order - scale), order);
//            subRow<<<dim3(blockX, 1, 1), dim3(threadX, 1, 1)>>>(matrix, mult, i, (*order + 1) * (*order - scale));
    __syncthreads();
//    cudaDeviceSynchronize();
}

__global__ void determinantByGauss(double *matrix, double *det, int *order) {
    int scale = *order;
    double first, mult;
    *det = 1;
    unsigned threadX, blockX;

//    __syncthreads();
    while (scale > 1) {
        if (scale > 1024) {
            threadX = 1024;
            blockX = 1 + (scale) / 1024;
        } else {
            threadX = scale;
            blockX = 1;
        }

        if (matrix[(*order + 1) * (*order - scale)] == 0) break;
        first = matrix[(*order + 1) * (*order - scale)];
        int count = (*order * *order - (*order + 1) * (*order - scale) + *order) / *order;
        abobaFunc<<<1, count>>>(matrix, first, *order, scale, blockX, threadX);
        __syncthreads();
        //        cudaDeviceSynchronize();
//        for (int i = (*order + 1) * (*order - scale) + *order; i < *order * *order; i += *order) {
//            mult = matrix[i] / first;
//            subRow<<<blockX, threadX>>>(matrix, mult, i, (*order + 1) * (*order - scale));
////            subRow<<<dim3(blockX, 1, 1), dim3(threadX, 1, 1)>>>(matrix, mult, i, (*order + 1) * (*order - scale));
//            cudaDeviceSynchronize();
//        }
        scale--;
    }

    for (unsigned i = *order; i >= 1; --i) *det *= matrix[(*order + 1) * (*order - i)];
//    __syncthreads();
}

void printMatrix(double *matrix, const int *SIZE) {
    for (int i = 0; i < *SIZE * *SIZE; ++i) {
        if (i % *SIZE == 0)printf("\n");
        printf("%f ", matrix[i]);
    }
    printf("\n");
}

void getResults() {
    FILE *f1, *f2;
    if ((f1 = fopen("../matrices.txt", "r")) == nullptr) {
        printf("Can't open file 'read.txt'\n");
        exit(-1);
    }
    if ((f2 = fopen("../results2.txt", "w")) == nullptr) {
        printf("Can't open file 'write.txt'\n");
        exit(-1);
    }
    double *matrix, det;
    int order;
    double *matrixCuda, *detCuda;
    int *orderCuda;
    clock_t startC, endC;
    cudaMalloc((void **) &detCuda, sizeof(double));
    cudaMalloc((void **) &orderCuda, sizeof(int));

    while (fscanf(f1, "%d", &order) == 1) {
        matrix = (double *) malloc(order * order * sizeof(double));
        if (!matrix) exit(-3);
        cudaMalloc((void **) &matrixCuda, order * order * sizeof(double));
        cudaMemcpy(orderCuda, &order, sizeof(int), cudaMemcpyHostToDevice);

        for (int i = 0; i < order * order; ++i) {
            fscanf(f1, "%lf", &matrix[i]);
        }
        cudaMemcpy(matrixCuda, matrix, order * order * sizeof(double), cudaMemcpyHostToDevice);

        startC = clock();
//        cudaEvent_t start, stop;
//        float elapsedTime;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);

//        cudaEventRecord(start, 0);
        determinantByGauss<<<1, 1>>>(matrixCuda, detCuda, orderCuda);
        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&elapsedTime, start, stop);
        endC = clock();
        cudaMemcpy(&det, detCuda, sizeof(double), cudaMemcpyDeviceToHost);
//        cudaMemcpy(matrix, matrixCuda, order*order*sizeof(double), cudaMemcpyDeviceToHost);
//        printMatrix(matrix, &order);
        fprintf(f2, "%ld %f\n", endC - startC, det);
        printf("\t %d \n", order);

        free(matrix);
        cudaFree(matrixCuda);
//        cudaEventDestroy(start);
//        cudaEventDestroy(stop);
    }
    cudaFree(detCuda);
    cudaFree(orderCuda);
    fclose(f1);
    fclose(f2);
}

int main() {
    getInfo();
    getResults();
    return 0;
}
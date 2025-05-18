
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <windows.h>

namespace fs = std::filesystem;

using namespace std;
using namespace std::chrono;

// Чтение матрицы из файла
vector<vector<int>> readMatrix(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        exit(1);
    }

    vector<vector<int>> matrix;
    string line;

    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        int value;

        while (ss >> value) {
            row.push_back(value);
        }

        matrix.push_back(row);
    }

    file.close();
    return matrix;
}

// Запись матрицы в файл
void writeMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return;
    }

    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            file << elem << " ";
        }
        file << endl;
    }

    file.close();
}

// CUDA kernel для умножения матриц
__global__ void matrixMulKernel(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Обёртка для вызова CUDA-ядра
vector<vector<int>> multiplyMatricesCUDA(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int N = A.size();

    size_t mem_size = N * N * sizeof(int);

    // Хост-память
    int* h_A = new int[N * N];
    int* h_B = new int[N * N];
    int* h_C = new int[N * N];

    // Копирование матриц в одномерный массив
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = A[i][j];
            h_B[i * N + j] = B[i][j];
        }

    // Устройство — GPU
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, mem_size);
    cudaMalloc(&d_B, mem_size);
    cudaMalloc(&d_C, mem_size);

    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);

    // Настройка сетки и блоков
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск ядра
    matrixMulKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // Ждём завершения ядра

    cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost);

    // Освобождение ресурсов
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;

    // Копируем в векторный результат
    vector<vector<int>> result(N, vector<int>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            result[i][j] = h_C[i * N + j];

    delete[] h_C;

    return result;
}

int main() {
    SetConsoleOutputCP(65001);

    int sizes[] = { 50, 100, 150, 250, 500, 1000 };

    fs::path resultsDir = fs::current_path().parent_path() / "results";
    if (!fs::exists(resultsDir)) fs::create_directories(resultsDir);

    for (int size : sizes) {
        string fileA = "../data/matrix_A_" + to_string(size) + ".txt";
        string fileB = "../data/matrix_B_" + to_string(size) + ".txt";
        string resultFile = resultsDir.string() + "/result_" + to_string(size) + ".txt";

        auto matrixA = readMatrix(fileA);
        auto matrixB = readMatrix(fileB);

        auto start = high_resolution_clock::now();
        auto resultMatrix = multiplyMatricesCUDA(matrixA, matrixB);
        auto end = high_resolution_clock::now();

        writeMatrix(resultFile, resultMatrix);

        auto duration = duration_cast<milliseconds>(end - start).count();
        cout << "Matrix size: " << size << "x" << size << endl;
        cout << "Execution time: " << duration << " ms" << endl;
        cout << "Task size: " << size * size << " elements" << endl;
        cout << endl;
    }

    return 0;
}
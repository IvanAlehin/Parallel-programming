#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <windows.h>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

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

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();

    vector<vector<int>> result(rowsA, vector<int>(colsB, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int k = 0; k < colsA; ++k) {
            int a_ik = A[i][k];  
            for (int j = 0; j < colsB; ++j) {
                result[i][j] += a_ik * B[k][j];  
            }
        }
    }

    return result;
}

int main() {
    SetConsoleOutputCP(65001);

    int sizes[] = { 50, 100, 150, 250, 500, 1000};

    fs::path resultsDir = fs::current_path().parent_path() / "results";

    if (!fs::exists(resultsDir)) fs::create_directories(resultsDir);

    for (int size : sizes) {
        string fileA = "../data/matrix_A_" + to_string(size) + ".txt";
        string fileB = "../data/matrix_B_" + to_string(size) + ".txt";
        string resultFile = resultsDir.string() + "/result_" + to_string(size) + ".txt";

        auto matrixA = readMatrix(fileA);
        auto matrixB = readMatrix(fileB);

        auto start = high_resolution_clock::now();
        auto resultMatrix = multiplyMatrices(matrixA, matrixB);
        auto end = high_resolution_clock::now();

        writeMatrix(resultFile, resultMatrix);

        auto duration = duration_cast<milliseconds>(end - start);
        cout << "Размер матрицы: " << size << "x" << size << endl;
        cout << "Время выполнения: " << duration.count() << " мс" << endl;

        int elements = size * size;
        cout << "Объем задачи: " << elements << " элементов" << endl;
        cout << endl;
    }

    return 0;
}

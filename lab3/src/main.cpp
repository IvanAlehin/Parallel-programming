#include <mpi.h>
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

ector<vector<int>> readMatrix(const string& filename, bool& success) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        success = false;
        return {};
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
    success = true;
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

vector<vector<int>> multiplyMatricesLocal(const vector<vector<int>>& A, const vector<vector<int>>& B) {
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SetConsoleOutputCP(65001);

    int sizes[] = { 50, 100, 150, 250, 500, 1000 };

    fs::path resultsDir = fs::current_path().parent_path() / "results";

    if (rank == 0) {
        if (!fs::exists(resultsDir)) {
            fs::create_directories(resultsDir);
        }
    }

    for (int size_matrix : sizes) {
        string fileA = "../data/matrix_A_" + to_string(size_matrix) + ".txt";
        string fileB = "../data/matrix_B_" + to_string(size_matrix) + ".txt";
        string resultFile = resultsDir.string() + "/result_" + to_string(size_matrix) + ".txt";

        vector<vector<int>> matrixA, matrixB;
        high_resolution_clock::time_point start, end;
        int valid_run = 1;

        if (rank == 0) {
            bool successA, successB;
            matrixA = readMatrix(fileA, successA);
            matrixB = readMatrix(fileB, successB);

            if (!successA || !successB) {
                valid_run = 0;
            }
            else if (matrixA[0].size() != matrixB.size()) {
                cerr << "Матрицы несовместимы для умножения!" << endl;
                valid_run = false;
            }

            start = high_resolution_clock::now();
        }

        MPI_Bcast(&valid_run, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid_run) {
            continue;
        }

        int rowsB = 0, colsB = 0;
        if (rank == 0) {
            rowsB = matrixB.size();
            colsB = matrixB[0].size();
        }
        MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> B_data;
        if (rank == 0) {
            B_data.resize(rowsB * colsB);
            for (int i = 0; i < rowsB; ++i) {
                for (int j = 0; j < colsB; ++j) {
                    B_data[i * colsB + j] = matrixB[i][j];
                }
            }
        }
        B_data.resize(rowsB * colsB);
        MPI_Bcast(B_data.data(), B_data.size(), MPI_INT, 0, MPI_COMM_WORLD);

        vector<vector<int>> local_B(rowsB, vector<int>(colsB));
        for (int i = 0; i < rowsB; ++i) {
            for (int j = 0; j < colsB; ++j) {
                local_B[i][j] = B_data[i * colsB + j];
            }
        }

        int rowsA = 0, colsA = 0;
        if (rank == 0) {
            rowsA = matrixA.size();
            colsA = matrixA[0].size();
        }
        MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> A_data;
        if (rank == 0) {
            A_data.resize(rowsA * colsA);
            for (int i = 0; i < rowsA; ++i) {
                for (int j = 0; j < colsA; ++j) {
                    A_data[i * colsA + j] = matrixA[i][j];
                }
            }
        }

        int block_size = rowsA / size;
        int extra = rowsA % size;
        int local_rows = (rank < extra) ? block_size + 1 : block_size;
        int local_A_size = local_rows * colsA;

        vector<int> sendcounts, displs;
        if (rank == 0) {
            sendcounts.resize(size);
            displs.resize(size);
            int current_disp = 0;
            for (int i = 0; i < size; ++i) {
                int chunk_size = (i < extra) ? block_size + 1 : block_size;
                sendcounts[i] = chunk_size * colsA;
                displs[i] = current_disp;
                current_disp += chunk_size * colsA;
            }
        }

        vector<int> local_A_data(local_A_size);
        MPI_Scatterv(A_data.data(), sendcounts.data(), displs.data(), MPI_INT,
            local_A_data.data(), local_A_size, MPI_INT, 0, MPI_COMM_WORLD);

        vector<vector<int>> local_A(local_rows, vector<int>(colsA));
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < colsA; ++j) {
                local_A[i][j] = local_A_data[i * colsA + j];
            }
        }

        vector<vector<int>> local_result = multiplyMatricesLocal(local_A, local_B);

        int cols_result = colsB;
        int local_result_size = local_rows * cols_result;
        vector<int> local_result_data(local_result_size);
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < cols_result; ++j) {
                local_result_data[i * cols_result + j] = local_result[i][j];
            }
        }

        vector<int> recvcounts, rdispls;
        vector<int> result_data;
        if (rank == 0) {
            recvcounts.resize(size);
            rdispls.resize(size);
            int current_disp = 0;
            for (int i = 0; i < size; ++i) {
                int chunk_size = (i < extra) ? block_size + 1 : block_size;
                recvcounts[i] = chunk_size * cols_result;
                rdispls[i] = current_disp;
                current_disp += chunk_size * cols_result;
            }
            result_data.resize(rowsA * cols_result);
        }

        MPI_Gatherv(local_result_data.data(), local_result_size, MPI_INT,
            result_data.data(), recvcounts.data(), rdispls.data(), MPI_INT,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            end = high_resolution_clock::now();

            vector<vector<int>> full_result(rowsA, vector<int>(cols_result));
            for (int i = 0; i < rowsA; ++i) {
                for (int j = 0; j < cols_result; ++j) {
                    full_result[i][j] = result_data[i * cols_result + j];
                }
            }

            writeMatrix(resultFile, full_result);

            auto duration = duration_cast<milliseconds>(end - start);
            cout << "Matrix size: " << size_matrix << "x" << size_matrix << endl;
            cout << "Execution time: " << duration.count() << " ms" << endl;
            int elements = size_matrix * size_matrix;
            cout << "Task size: " << elements << " elements" << endl;
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

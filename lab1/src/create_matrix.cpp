#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <windows.h>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

using namespace std;

void createMatrix(const string& filename, int rows, int cols, int seed) {
    
    fs::path project_dir = fs::current_path().parent_path();
    fs::path data_dir = project_dir / "data";  

    fs::create_directories(data_dir);

    ofstream file(data_dir / filename);
    if (!file.is_open()) {
        cerr << "Не удалось открыть файл для записи: " << filename << endl;
        return;
    }

    std::default_random_engine generator(seed);  
    std::uniform_int_distribution<int> distribution(-99, 99); 

    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << distribution(generator) << " ";  
        }
        file << endl;
    }

    file.close();
}

int main() {
    SetConsoleOutputCP(65001);

    int sizes[] = { 50, 100, 150, 250, 500, 1000, 2500, 5000 };

    int seedA = static_cast<int>(time(0)); 
    int seedB = seedA + 1;  

    for (int size : sizes) {
        string fileA = "matrix_A_" + to_string(size) + ".txt";
        string fileB = "matrix_B_" + to_string(size) + ".txt";

        createMatrix(fileA, size, size, seedA);
        createMatrix(fileB, size, size, seedB);

        cout << "Генерация матриц " << fileA << " и " << fileB << " завершена." << endl;

        seedA++;
        seedB++;
    }

    return 0;
}

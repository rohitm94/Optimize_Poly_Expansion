#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " n degree" << std::endl;
        return -1;
    }

    int n = atoi(argv[1]); //TODO: atoi is an unsafe function
    int degree = atoi(argv[2]);
    int nbiter = atoi(argv[3]);

    float *array ;
    cudaMallocHost((void **)&array, n * sizeof(float));
    float *poly = new float[degree + 1];
    for (int i = 0; i < n; ++i)
        array[i] = 1.;

    for (int i = 0; i < degree + 1; ++i)
        poly[i] = 1.;

    float *d_array;

    cudaMalloc((void **)&d_array, n * sizeof(float));

    std::chrono::time_point<std::chrono::system_clock> begin, end;
    begin = std::chrono::system_clock::now();

    for (int iter = 0; iter < nbiter; ++iter)
        cudaMemcpy(d_array, array, n * sizeof(float), cudaMemcpyHostToDevice);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> totaltime = (end - begin);

    cudaFree(d_array);
    cudaFreeHost(array);

    std::cout << "Latency of "<< nbiter <<"times is " << totaltime.count() << std::endl;
    std::cout << "Latency of PCI express is " << totaltime.count() / nbiter << std::endl;

    delete[] array;
    delete[] poly;

    return 0;
}
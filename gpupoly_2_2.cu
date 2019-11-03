#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{


    int n = atoi(argv[1]); //TODO: atoi is an unsafe function
    int nbiter = atoi(argv[2]);

    float *array = new float[n];
    for(int i = 0; i < n; ++i)
        array[i] = 1.;
    float *d_array;

    cudaMalloc((void **)&d_array, n * sizeof(float));

    std::chrono::time_point<std::chrono::system_clock> begin, end;
    begin = std::chrono::system_clock::now();

    for(int iter = 0; iter < nbiter; ++iter)
        cudaMemcpy(d_array, array, n * sizeof(float), cudaMemcpyHostToDevice);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> totaltime = (end - begin);

    cudaFree(d_array);

    std::cout << n*sizeof(float)/1000 <<" "<< (n*sizeof(float))/(totaltime.count()*nbiter) << std::endl;

    delete[] array;

    return 0;
}

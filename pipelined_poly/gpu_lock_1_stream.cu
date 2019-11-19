#include <iostream>
#include <chrono>

#define BLOCKSIZE 256

__global__ void polynomial_expansion(float *poly, int degree, int n, float *array)
{
    //TODO: Write code to use the GPU here!
    //code should write the output back to array
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        float temp = array[index];
        float out = 0, xtothepowerof = 1;
        for (int i = 0; i <= degree; i++)
        {
            out += xtothepowerof * poly[i];
            xtothepowerof *= temp;
        }
        array[index] = out;
    }
}

int main(int argc, char *argv[])
{
    //TODO: add usage

    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " n degree" << std::endl;
        return -1;
    }

    int n = atoi(argv[1]); //TODO: atoi is an unsafe function
    int degree = atoi(argv[2]);
    int nbiter = 1;

    float* array = NULL;
    float* poly = NULL;
    
    cudaMallocHost((void **)&array,sizeof(float)*n);
    cudaMallocHost((void **)&poly,sizeof(float)*(degree+1));
    for (int i = 0; i < n; ++i)
        array[i] = 1.;

    for (int i = 0; i < degree + 1; ++i)
        poly[i] = 1.;

    float *d_array, *d_poly;

    cudaMalloc((void **)&d_array, n * sizeof(float));
    cudaMalloc((void **)&d_poly, (degree + 1) * sizeof(float));

    int size = n * sizeof(float) / 1;

    cudaStream_t stream[1];
    for (int i = 0; i < 1; ++i)
        cudaStreamCreate(&stream[i]);

    std::chrono::time_point<std::chrono::system_clock> begin, end;
    begin = std::chrono::system_clock::now();

    for (int i = 0; i < 1; ++i) {
        cudaMemcpyAsync(d_array, array ,size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_poly, poly, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        polynomial_expansion <<<(n + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream[i]>>>(d_poly, degree, n, d_array);
        cudaMemcpyAsync(array, d_array,size, cudaMemcpyDeviceToHost, stream[i]);
        }
    /*cudaMemcpy(d_array, array, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poly, poly, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);

    polynomial_expansion<<<(n + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_poly, degree, n, d_array);*/

    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> totaltime = (end - begin);
    //cudaMemcpy(array, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_poly);

    std::cerr << array[0] << std::endl;
    std::cout << n*sizeof(float)/1000 << " " << totaltime.count() << " " << ((n+degree+1)*sizeof(float)*nbiter)/totaltime.count() << std::endl;

    delete[] array;
    delete[] poly;

    return 0;
}
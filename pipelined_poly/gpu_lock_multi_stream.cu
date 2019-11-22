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

    char *ptr;
    long long int n = strtol(argv[1],&ptr,10);
    int degree = atoi(argv[2]);
    int nbiter = 2;

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
    
        long long int size = n * sizeof(float) / 4;
    
        cudaStream_t stream[4];
        for (int i = 0; i < 4; ++i){
            cudaStreamCreate(&stream[i]);
        }
    
    
        cudaMemcpyAsync(d_poly, poly, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    
        std::chrono::time_point<std::chrono::system_clock> begin, end;
        begin = std::chrono::system_clock::now();
        for(int k = 1; k <=nbiter; k++){
            for (int i = 0; i < 4; ++i) {
                cudaMemcpyAsync(d_array+ i*size, array + i*size,size, cudaMemcpyHostToDevice, stream[i]);
                polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream[i]>>>(d_poly, degree, n/4, d_array + i*size);
                cudaMemcpyAsync(array+ i*size, d_array+ i*size,size, cudaMemcpyDeviceToHost, stream[i]);
                }
            }
    
        cudaDeviceSynchronize();
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> totaltime = (end - begin);
    
        for (int i = 0; i < 4; ++i){
            cudaStreamDestroy(stream[i]);
        }
        cudaFree(d_array);
        cudaFree(d_poly);

        double pciBW = 1.50e+10, gpumemBW = 2.88e+11 , gpuflopRate = 1.73e+12 , pciLat = 8.80597e-06;

        double HtD = pciLat + (((n)*(sizeof(float)))/pciBW);
        double DtH = pciLat + (((n)*(sizeof(float)))/pciBW);

        double dProc = std::max((3.0*(n)*(degree+1)/(gpuflopRate)),((sizeof(float)*(n+degree+1)/(gpumemBW))));
 
        double ideal_time = std::max(dProc,(HtD+DtH));
        
        std::cout << n*sizeof(float)<< " " << array[0]<< " " << degree << " " << totaltime.count() << " " << ((n))/ideal_time << " " << ((n)*nbiter)/totaltime.count() << std::endl;
    
        cudaFreeHost(array);
        cudaFreeHost(poly);
    
        return 0;
    }
    
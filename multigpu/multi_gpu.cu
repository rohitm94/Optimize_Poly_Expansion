#include <iostream>
#include <chrono>

#define BLOCKSIZE 512

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

    cudaMallocHost((void **)&array,sizeof(float)*(n/2));
    cudaMallocHost((void **)&poly,sizeof(float)*(degree+1));
    for (int i = 0; i < n/2; ++i)
        array[i] = 1.;

    for (int i = 0; i < (degree + 1); ++i)
        poly[i] = 1.;

    float *d_array, *d_poly;

    cudaMalloc((void **)&d_array, (n/2) * sizeof(float));
    cudaMalloc((void **)&d_poly, (degree + 1) * sizeof(float));

    long long int size = n * sizeof(float) / 8;


    cudaStream_t stream[4];
    for (int i = 0; i < 4; ++i){
        cudaStreamCreate(&stream[i]);
    }



    cudaMemcpyAsync(d_poly, poly, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaSetDevice(1);
    float* array1 = NULL;
    float* poly1 = NULL;

    cudaMallocHost((void **)&array1,sizeof(float)*(n/2));
    cudaMallocHost((void **)&poly1,sizeof(float)*(degree+1));
    for (int i = 0; i < n/2; ++i)
        array1[i] = 1.;

    for (int i = 0; i < (degree + 1); ++i)
        poly1[i] = 1.;

    float *d_array1, *d_poly1;

    cudaMalloc((void **)&d_array1, (n/2) * sizeof(float));
    cudaMalloc((void **)&d_poly1, (degree + 1) * sizeof(float));


    cudaStream_t stream1[4];
    for (int i = 0; i < 4; ++i){
        cudaStreamCreate(&stream1[i]);
    }



    cudaMemcpyAsync(d_poly1, poly1, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


        std::chrono::time_point<std::chrono::system_clock> begin, end;
        begin = std::chrono::system_clock::now();
        for(int k = 1; k <=nbiter; k++){
            cudaSetDevice(0);
            for (int i = 0; i < 4; ++i) {
                cudaMemcpyAsync(d_array+ i*size, array + i*size,size, cudaMemcpyHostToDevice, stream[i]);
                }
            for (int i = 0; i < 4; ++i) {
                 polynomial_expansion <<<((n/8) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream[i]>>>(d_poly, degree, n/8, d_array + i*size);
                cudaMemcpyAsync(array+ i*size, d_array+ i*size,size, cudaMemcpyDeviceToHost, stream[i]);
                }
            cudaSetDevice(1);
            for (int i = 0; i < 4; ++i) {
                cudaMemcpyAsync(d_array1+ i*size, array1 + i*size,size, cudaMemcpyHostToDevice, stream1[i]);
                }
            for (int i = 0; i < 4; ++i) {
                polynomial_expansion <<<((n/8) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream1[i]>>>(d_poly, degree, n/8, d_array1 + i*size);
                cudaMemcpyAsync(array1+ i*size, d_array1+ i*size,size, cudaMemcpyDeviceToHost, stream1[i]);
                }
            }

        cudaDeviceSynchronize();
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> totaltime = (end - begin);

        cudaSetDevice(0);
        for (int i = 0; i < 4; ++i){
            cudaStreamDestroy(stream[i]);
        }
        cudaFree(d_array);
        cudaFree(d_poly);
        cudaSetDevice(1);
        for (int i = 0; i < 4; ++i){
            cudaStreamDestroy(stream1[i]);
        }
        cudaFree(d_array1);
        cudaFree(d_poly1);

        double pciBW = 1.50e+10, gpumemBW = 2.88e+11 , gpuflopRate = 1.43e+12 , pciLat = 8.80594e-06;

        double HtD = pciLat + double(((nbiter*(n/2))*(sizeof(float)))/pciBW);
        std::cout<<"HTD: "<<HtD<<std::endl;
        double DtH = pciLat + double(((nbiter*(n/2))*(sizeof(float)))/pciBW);

        double dProc = std::max(double((3.0*(n/2)*(degree+1))/(gpuflopRate)),(double(sizeof(float)*((nbiter*(n/2))+degree+1))/(gpumemBW)));
        std::cout<<"dproc:"<<dProc<<" "<<double((3.0*(n/2)*(degree+1))/(gpuflopRate))<<" "<<(double(sizeof(float)*((nbiter*(n/2))+degree+1))/(gpumemBW))<<std::endl;
        double ideal_time = std::max(dProc,2*(HtD+DtH));

        std::cout << double(n*sizeof(float))<< " " << degree << " " << ideal_time << " " << totaltime.count() << " " << (double(n*(degree+1)))/(ideal_time) << " " << ((n)*nbiter)/totaltime.count() << std::endl;
        cudaSetDevice(0);
        cudaFreeHost(array);
        cudaFreeHost(poly);
        cudaSetDevice(1);
        cudaFreeHost(array1);
        cudaFreeHost(poly1);

        return 0;
    }


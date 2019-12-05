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

    long long int size = n * sizeof(float) / 4;




    cudaMemcpy(d_poly, poly, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);

        std::chrono::time_point<std::chrono::system_clock> begin, end;
        begin = std::chrono::system_clock::now();
        for(int k = 1; k <=nbiter; k++){
            cudaStream_t stream[4];
            for (int i = 0; i < 4; ++i){
                cudaStreamCreate(&stream[i]);
            }
            //for (int i = 0; i < 4; ++i) {
                cudaMemcpyAsync(d_array+ 0*size, array + 0*size,size, cudaMemcpyHostToDevice, stream[0]);
                polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream[0]>>>(d_poly, degree, n/4, d_array + 0*size);
                cudaMemcpyAsync(array+ 0*size, d_array+ 0*size,size, cudaMemcpyDeviceToHost, stream[0]);

                cudaMemcpyAsync(d_array+ 1*size, array + 1*size,size, cudaMemcpyHostToDevice, stream[1]);
                polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream[1]>>>(d_poly, degree, n/4, d_array + 1*size);
                cudaMemcpyAsync(array+ 1*size, d_array+ 1*size,size, cudaMemcpyDeviceToHost, stream[1]);

                cudaMemcpyAsync(d_array+ 2*size, array + 2*size,size, cudaMemcpyHostToDevice, stream[2]);
                polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream[2]>>>(d_poly, degree, n/4, d_array + 2*size);
                cudaMemcpyAsync(array+ 2*size, d_array+ 2*size,size, cudaMemcpyDeviceToHost, stream[2]);

                cudaMemcpyAsync(d_array+ 3*size, array + 3*size,size, cudaMemcpyHostToDevice, stream[3]);
                polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream[3]>>>(d_poly, degree, n/4, d_array + 3*size);
                cudaMemcpyAsync(array+ 3*size, d_array+ 3*size,size, cudaMemcpyDeviceToHost, stream[3]);
                //}
                cudaStreamSynchronize(stream[0]); 
                cudaStreamSynchronize(stream[1]);
                cudaStreamSynchronize(stream[2]);
                cudaStreamSynchronize(stream[3]);
            for (int i = 0; i < 4; ++i){
                cudaStreamDestroy(stream[i]);
            }
            }

        //cudaDeviceSynchronize();
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> totaltime = (end - begin);


        cudaFree(d_array);
        cudaFree(d_poly);

        double pciBW = 1.50e+10, gpumemBW = 2.88e+11 , gpuflopRate = 1.43e+12 , pciLat = 8.80594e-06;

        double HtD =  double(((nbiter*n)*(sizeof(float)))/pciBW);
        double DtH =  double(((nbiter*n)*(sizeof(float)))/pciBW);

        double dProc = std::max(double((3.0*(n)*(degree+1))/(gpuflopRate)),(double(sizeof(float)*((nbiter*n)+degree+1))/(gpumemBW)));

        double ideal_time = std::max(dProc,(HtD+DtH));

        std::cout << n*sizeof(float)<< " " << degree << " " << ideal_time << " " << totaltime.count() << " " << (n*(degree+1))/(ideal_time) << " " << ((n*(degree+1))*nbiter)/totaltime.count() << std::endl;

        cudaFreeHost(array);
        cudaFreeHost(poly);

        return 0;
    }


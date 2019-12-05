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


    /*cudaStream_t stream[4];
    for (int i = 0; i < 4; ++i){
        cudaStreamCreate(&stream[i]);
    }*/



    cudaMemcpy(d_poly, poly, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);

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


    /*cudaStream_t stream1[4];
    for (int i = 0; i < 4; ++i){
        cudaStreamCreate(&stream1[i]);
    }*/



    cudaMemcpy(d_poly1, poly1, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);


        std::chrono::time_point<std::chrono::system_clock> begin, end;
        begin = std::chrono::system_clock::now();
        for(int k = 1; k <=nbiter; k++){
            cudaSetDevice(0);
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

            cudaSetDevice(1);
            cudaStream_t stream1[4];
            for (int i = 0; i < 4; ++i){
                cudaStreamCreate(&stream1[i]);
            }
            cudaMemcpyAsync(d_array1+ 0*size, array1 + 0*size,size, cudaMemcpyHostToDevice, stream1[0]);
            polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream1[0]>>>(d_poly1, degree, n/4, d_array1 + 0*size);
            cudaMemcpyAsync(array1+ 0*size, d_array1+ 0*size,size, cudaMemcpyDeviceToHost, stream1[0]);

            cudaMemcpyAsync(d_array1+ 1*size, array1 + 1*size,size, cudaMemcpyHostToDevice, stream1[1]);
            polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream1[1]>>>(d_poly1, degree, n/4, d_array1 + 1*size);
            cudaMemcpyAsync(array1+ 1*size, d_array1+ 1*size,size, cudaMemcpyDeviceToHost, stream1[1]);

            cudaMemcpyAsync(d_array1+ 2*size, array1 + 2*size,size, cudaMemcpyHostToDevice, stream1[2]);
            polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream1[2]>>>(d_poly1, degree, n/4, d_array1 + 2*size);
            cudaMemcpyAsync(array1+ 2*size, d_array1+ 2*size,size, cudaMemcpyDeviceToHost, stream1[2]);

            cudaMemcpyAsync(d_array1 + 3*size, array1 + 3*size,size, cudaMemcpyHostToDevice, stream1[3]);
            polynomial_expansion <<<((n/4) + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream1[3]>>>(d_poly1, degree, n/4, d_array1 + 3*size);
            cudaMemcpyAsync(array1 + 3*size, d_array1 + 3*size,size, cudaMemcpyDeviceToHost, stream1[3]);
            //}
            cudaStreamSynchronize(stream1[0]); 
            cudaStreamSynchronize(stream1[1]);
            cudaStreamSynchronize(stream1[2]);
            cudaStreamSynchronize(stream1[3]);
            for (int i = 0; i < 4; ++i){
                cudaStreamDestroy(stream1[i]);
            }
            cudaSetDevice(0);
            cudaStreamSynchronize(stream[0]); 
            cudaStreamSynchronize(stream[1]);
            cudaStreamSynchronize(stream[2]);
            cudaStreamSynchronize(stream[3]);
            for (int i = 0; i < 4; ++i){
                cudaStreamDestroy(stream[i]);
            }
            }

        cudaDeviceSynchronize();
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> totaltime = (end - begin);

        /*cudaSetDevice(0);
        for (int i = 0; i < 4; ++i){
            cudaStreamDestroy(stream[i]);
        }*/
        cudaSetDevice(0);
        cudaFreeHost(array);
        cudaFreeHost(poly);
        cudaFree(d_array);
        cudaFree(d_poly);
        /*cudaSetDevice(1);
        for (int i = 0; i < 4; ++i){
            cudaStreamDestroy(stream1[i]);
        }*/
        cudaSetDevice(1);
        cudaFreeHost(array1);
        cudaFreeHost(poly1);
        cudaFree(d_array1);
        cudaFree(d_poly1);

        double pciBW = 1.50e+10, gpumemBW = 2.88e+11 , gpuflopRate = 1.73e+12 , pciLat = 8.80594e-06;

        double HtD = double(((nbiter*(n/2))*(sizeof(float)))/pciBW);
        //std::cout<<"HTD: "<<HtD<<std::endl;
        double DtH = double(((nbiter*(n/2))*(sizeof(float)))/pciBW);

        double dProc = std::max(double((3.0*(n/2)*(degree+1))/(gpuflopRate)),(double(sizeof(float)*((nbiter*(n/2))+degree+1))/(gpumemBW)));
        //std::cout<<"dproc:"<<dProc<<" "<<double((3.0*(n/2)*(degree+1))/(gpuflopRate))<<" "<<(double(sizeof(float)*((nbiter*(n/2))+degree+1))/(gpumemBW))<<std::endl;
        double ideal_time = std::max(dProc,(HtD+DtH));

        std::cout << double(n*sizeof(float))<< " " << degree << " " << ideal_time << " " << totaltime.count() << " " << (double(n)/(ideal_time)) << " " << ((n)*nbiter)/totaltime.count() << std::endl;
        

        return 0;
    }


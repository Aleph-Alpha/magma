
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

constexpr uint32_t THREADS_PER_BLOCK = 512;


// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_0||X| + |b_1||X|^2 + ... + |b_i||X|^{i+1}


template <typename scalar_t>
__global__ void rational_cuda_forward_A_kernel_5_4( const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    
    scalar_t a_0 = a[0];
    
    scalar_t a_1 = a[1];
    
    scalar_t a_2 = a[2];
    
    scalar_t a_3 = a[3];
    
    scalar_t a_4 = a[4];
    
    scalar_t a_5 = a[5];
    
    
    scalar_t ab_0 = abs(b[0]);
    
    scalar_t ab_1 = abs(b[1]);
    
    scalar_t ab_2 = abs(b[2]);
    
    scalar_t ab_3 = abs(b[3]);
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
        
        
        scalar_t axp1 = abs(xp1);
        
        scalar_t axp2 = abs(xp2);
        
        scalar_t axp3 = abs(xp3);
        
        scalar_t axp4 = abs(xp4);
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
                ;

        scalar_t Q = scalar_t(1.0)
                + ab_0 * axp1
                + ab_1 * axp2
                + ab_2 * axp3
                + ab_3 * axp4
                ;
        // if (!index%1000){
        // printf("Q: %d\n",Q );
        // printf("P: %d\n",P );
        // printf("res: %d\n", P/Q);

        // }
        result[index] = P / Q;
    }
}


at::Tensor rational_cuda_forward_A_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_A_5_4", ([&] {
    rational_cuda_forward_A_kernel_5_4<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));


    return result;
}


//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |b_0||X| + |b_1||X|^2 + |b_2||X|^3
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//S(X) = sign(X) * ( |b_0| + 2|b_1||X| + 3|b_2||X|^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,5}
//dF/db_i = (-P(X)/Q(X)^2) * sign(b_i) * |X^{i+1}| , i \in {0,4}

template <typename scalar_t>
__global__ void rational_cuda_backward_A_kernel_5_4(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[6];
    __shared__ double sdb[4];

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
            }

    __syncthreads();
    
    scalar_t d_a0 = 0;
    scalar_t a_0 = a[0];
    
    scalar_t d_a1 = 0;
    scalar_t a_1 = a[1];
    
    scalar_t d_a2 = 0;
    scalar_t a_2 = a[2];
    
    scalar_t d_a3 = 0;
    scalar_t a_3 = a[3];
    
    scalar_t d_a4 = 0;
    scalar_t a_4 = a[4];
    
    scalar_t d_a5 = 0;
    scalar_t a_5 = a[5];
    
    
    scalar_t d_b0 = 0;
    scalar_t b_0 = b[0];
    scalar_t ab_0 = abs(b_0);
    
    scalar_t d_b1 = 0;
    scalar_t b_1 = b[1];
    scalar_t ab_1 = abs(b_1);
    
    scalar_t d_b2 = 0;
    scalar_t b_2 = b[2];
    scalar_t ab_2 = abs(b_2);
    
    scalar_t d_b3 = 0;
    scalar_t b_3 = b[3];
    scalar_t ab_3 = abs(b_3);
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {
        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

                scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
                scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
                scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
                scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
                ;

        scalar_t Q = scalar_t(1.0)
                + ab_0 * axp1
                + ab_1 * axp2
                + ab_2 * axp3
                + ab_3 * axp4
                ;

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                ;

        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ab_0

                + scalar_t(2.0) * ab_1 * axp1
                + scalar_t(3.0) * ab_2 * axp2
                + scalar_t(4.0) * ab_3 * axp3
                );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

                scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), b_0 ) * axp1;
        d_b0 += d_i_b0 * grad_o;
                scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), b_1 ) * axp2;
        d_b1 += d_i_b1 * grad_o;
                scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), b_2 ) * axp3;
        d_b2 += d_i_b2 * grad_o;
                scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), b_3 ) * axp4;
        d_b3 += d_i_b3 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

        
        scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
        
        scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
        
        scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
        
        scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
        
        scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
            }
}


std::vector<torch::Tensor> rational_cuda_backward_A_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_A_5_4", ([&] {
    rational_cuda_backward_A_kernel_5_4<scalar_t>
        <<<16, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}



// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_0||X| + |b_1||X|^2 + ... + |b_i||X|^{i+1}


template <typename scalar_t>
__global__ void rational_cuda_forward_A_kernel_7_6( const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    
    scalar_t a_0 = a[0];
    
    scalar_t a_1 = a[1];
    
    scalar_t a_2 = a[2];
    
    scalar_t a_3 = a[3];
    
    scalar_t a_4 = a[4];
    
    scalar_t a_5 = a[5];
    
    scalar_t a_6 = a[6];
    
    scalar_t a_7 = a[7];
    
    
    scalar_t ab_0 = abs(b[0]);
    
    scalar_t ab_1 = abs(b[1]);
    
    scalar_t ab_2 = abs(b[2]);
    
    scalar_t ab_3 = abs(b[3]);
    
    scalar_t ab_4 = abs(b[4]);
    
    scalar_t ab_5 = abs(b[5]);
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
                scalar_t xp6 = xp5 * xp1;
                scalar_t xp7 = xp6 * xp1;
        
        
        scalar_t axp1 = abs(xp1);
        
        scalar_t axp2 = abs(xp2);
        
        scalar_t axp3 = abs(xp3);
        
        scalar_t axp4 = abs(xp4);
        
        scalar_t axp5 = abs(xp5);
        
        scalar_t axp6 = abs(xp6);
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
        
        + a_6 * xp6
        
        + a_7 * xp7
                ;

        scalar_t Q = scalar_t(1.0)
                + ab_0 * axp1
                + ab_1 * axp2
                + ab_2 * axp3
                + ab_3 * axp4
                + ab_4 * axp5
                + ab_5 * axp6
                ;
        // if (!index%1000){
        // printf("Q: %d\n",Q );
        // printf("P: %d\n",P );
        // printf("res: %d\n", P/Q);

        // }
        result[index] = P / Q;
    }
}


at::Tensor rational_cuda_forward_A_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_A_7_6", ([&] {
    rational_cuda_forward_A_kernel_7_6<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));


    return result;
}


//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |b_0||X| + |b_1||X|^2 + |b_2||X|^3
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//S(X) = sign(X) * ( |b_0| + 2|b_1||X| + 3|b_2||X|^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,7}
//dF/db_i = (-P(X)/Q(X)^2) * sign(b_i) * |X^{i+1}| , i \in {0,6}

template <typename scalar_t>
__global__ void rational_cuda_backward_A_kernel_7_6(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[8];
    __shared__ double sdb[6];

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
        
        sda[6] = 0;
        
        sda[7] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
        
        sdb[4] = 0;
        
        sdb[5] = 0;
            }

    __syncthreads();
    
    scalar_t d_a0 = 0;
    scalar_t a_0 = a[0];
    
    scalar_t d_a1 = 0;
    scalar_t a_1 = a[1];
    
    scalar_t d_a2 = 0;
    scalar_t a_2 = a[2];
    
    scalar_t d_a3 = 0;
    scalar_t a_3 = a[3];
    
    scalar_t d_a4 = 0;
    scalar_t a_4 = a[4];
    
    scalar_t d_a5 = 0;
    scalar_t a_5 = a[5];
    
    scalar_t d_a6 = 0;
    scalar_t a_6 = a[6];
    
    scalar_t d_a7 = 0;
    scalar_t a_7 = a[7];
    
    
    scalar_t d_b0 = 0;
    scalar_t b_0 = b[0];
    scalar_t ab_0 = abs(b_0);
    
    scalar_t d_b1 = 0;
    scalar_t b_1 = b[1];
    scalar_t ab_1 = abs(b_1);
    
    scalar_t d_b2 = 0;
    scalar_t b_2 = b[2];
    scalar_t ab_2 = abs(b_2);
    
    scalar_t d_b3 = 0;
    scalar_t b_3 = b[3];
    scalar_t ab_3 = abs(b_3);
    
    scalar_t d_b4 = 0;
    scalar_t b_4 = b[4];
    scalar_t ab_4 = abs(b_4);
    
    scalar_t d_b5 = 0;
    scalar_t b_5 = b[5];
    scalar_t ab_5 = abs(b_5);
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {
        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

                scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
                scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
                scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
                scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
                scalar_t xp6 = xp5 * xp1;
        scalar_t axp6 = abs(xp6);
                scalar_t xp7 = xp6 * xp1;
        scalar_t axp7 = abs(xp7);
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
        
        + a_6*xp6
        
        + a_7*xp7
                ;

        scalar_t Q = scalar_t(1.0)
                + ab_0 * axp1
                + ab_1 * axp2
                + ab_2 * axp3
                + ab_3 * axp4
                + ab_4 * axp5
                + ab_5 * axp6
                ;

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                + scalar_t(6.0) * a_6 * xp5
                + scalar_t(7.0) * a_7 * xp6
                ;

        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ab_0

                + scalar_t(2.0) * ab_1 * axp1
                + scalar_t(3.0) * ab_2 * axp2
                + scalar_t(4.0) * ab_3 * axp3
                + scalar_t(5.0) * ab_4 * axp4
                + scalar_t(6.0) * ab_5 * axp5
                );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

                scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), b_0 ) * axp1;
        d_b0 += d_i_b0 * grad_o;
                scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), b_1 ) * axp2;
        d_b1 += d_i_b1 * grad_o;
                scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), b_2 ) * axp3;
        d_b2 += d_i_b2 * grad_o;
                scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), b_3 ) * axp4;
        d_b3 += d_i_b3 * grad_o;
                scalar_t d_i_b4 = mpq2 * copysign( scalar_t(1.0), b_4 ) * axp5;
        d_b4 += d_i_b4 * grad_o;
                scalar_t d_i_b5 = mpq2 * copysign( scalar_t(1.0), b_5 ) * axp6;
        d_b5 += d_i_b5 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

        
        scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
        
        scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
        
        scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
        
        scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
        
        scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
        
        scalar_t d_i_a6  = xp6/Q;
        d_a6 += d_i_a6 * grad_o;
        
        scalar_t d_i_a7  = xp7/Q;
        d_a7 += d_i_a7 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
    
    atomicAdd(&sda[6], d_a6);
    
    atomicAdd(&sda[7], d_a7);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    atomicAdd(&sdb[4], d_b4);
    
    atomicAdd(&sdb[5], d_b5);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
        
        atomicAdd(&d_a[6], sda[6]);
        
        atomicAdd(&d_a[7], sda[7]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
        
        atomicAdd(&d_b[4], sdb[4]);
        
        atomicAdd(&d_b[5], sdb[5]);
            }
}


std::vector<torch::Tensor> rational_cuda_backward_A_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_A_7_6", ([&] {
    rational_cuda_backward_A_kernel_7_6<scalar_t>
        <<<16, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}



// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_1*X + b_2*X^2 + ... + b_n*X^n|



template <typename scalar_t>
__global__ void rational_cuda_forward_B_kernel_5_4( const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    
    scalar_t a_0 = a[0];
    
    scalar_t a_1 = a[1];
    
    scalar_t a_2 = a[2];
    
    scalar_t a_3 = a[3];
    
    scalar_t a_4 = a[4];
    
    scalar_t a_5 = a[5];
    
    
    scalar_t b_0 = b[0];
    
    scalar_t b_1 = b[1];
    
    scalar_t b_2 = b[2];
    
    scalar_t b_3 = b[3];
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
                ;

        scalar_t Q = scalar_t(1.0) + abs(
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                );

        result[index] = P/Q;
    }
}


at::Tensor rational_cuda_forward_B_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_B_5_4", ([&] {
    rational_cuda_forward_B_kernel_5_4<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));

    return result;
}




//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |A(X)|
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//A(X) = b_0*X + b_1*X^2 + b_2*X^3
//S(X) = sign(A(X)) * ( b_0 + 2*b_1*X + 3*b_2*X^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,5}
//dF/db_i = (-P(X)/Q(X)^2) * sign(A(X)) * X^{i+1} , i \in {0,4}


template <typename scalar_t>
__global__ void rational_cuda_backward_B_kernel_5_4(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[6];
    __shared__ double sdb[4];

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
            }

    __syncthreads();
    
    scalar_t d_a0 = 0;
    scalar_t a_0 = a[0];
    
    scalar_t d_a1 = 0;
    scalar_t a_1 = a[1];
    
    scalar_t d_a2 = 0;
    scalar_t a_2 = a[2];
    
    scalar_t d_a3 = 0;
    scalar_t a_3 = a[3];
    
    scalar_t d_a4 = 0;
    scalar_t a_4 = a[4];
    
    scalar_t d_a5 = 0;
    scalar_t a_5 = a[5];
    
    
    scalar_t d_b0 = 0;
    scalar_t b_0 = b[0];
    
    scalar_t d_b1 = 0;
    scalar_t b_1 = b[1];
    
    scalar_t d_b2 = 0;
    scalar_t b_2 = b[2];
    
    scalar_t d_b3 = 0;
    scalar_t b_3 = b[3];
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {
        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
                ;

        scalar_t A =
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                ;

        scalar_t Q = scalar_t(1.0) + abs(A);

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                ;

        scalar_t S = copysign( scalar_t(1.0), A ) * (b_0

                + scalar_t(2.0) * b_1 * xp1
                + scalar_t(3.0) * b_2 * xp2
                + scalar_t(4.0) * b_3 * xp3
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

                scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), A ) * xp1;
        d_b0 += d_i_b0 * grad_o;
                scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), A ) * xp2;
        d_b1 += d_i_b1 * grad_o;
                scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), A ) * xp3;
        d_b2 += d_i_b2 * grad_o;
                scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), A ) * xp4;
        d_b3 += d_i_b3 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

        
        scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
        
        scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
        
        scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
        
        scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
        
        scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
            }
}




std::vector<torch::Tensor> rational_cuda_backward_B_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_B_5_4", ([&] {
    rational_cuda_backward_B_kernel_5_4<scalar_t>
        <<<16, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}




// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_1*X + b_2*X^2 + ... + b_n*X^n|



template <typename scalar_t>
__global__ void rational_cuda_forward_B_kernel_7_6( const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    
    scalar_t a_0 = a[0];
    
    scalar_t a_1 = a[1];
    
    scalar_t a_2 = a[2];
    
    scalar_t a_3 = a[3];
    
    scalar_t a_4 = a[4];
    
    scalar_t a_5 = a[5];
    
    scalar_t a_6 = a[6];
    
    scalar_t a_7 = a[7];
    
    
    scalar_t b_0 = b[0];
    
    scalar_t b_1 = b[1];
    
    scalar_t b_2 = b[2];
    
    scalar_t b_3 = b[3];
    
    scalar_t b_4 = b[4];
    
    scalar_t b_5 = b[5];
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
                scalar_t xp6 = xp5 * xp1;
                scalar_t xp7 = xp6 * xp1;
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
        
        + a_6 * xp6
        
        + a_7 * xp7
                ;

        scalar_t Q = scalar_t(1.0) + abs(
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                + b_4 * xp5
                + b_5 * xp6
                );

        result[index] = P/Q;
    }
}


at::Tensor rational_cuda_forward_B_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_B_7_6", ([&] {
    rational_cuda_forward_B_kernel_7_6<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));

    return result;
}




//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |A(X)|
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//A(X) = b_0*X + b_1*X^2 + b_2*X^3
//S(X) = sign(A(X)) * ( b_0 + 2*b_1*X + 3*b_2*X^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,7}
//dF/db_i = (-P(X)/Q(X)^2) * sign(A(X)) * X^{i+1} , i \in {0,6}


template <typename scalar_t>
__global__ void rational_cuda_backward_B_kernel_7_6(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[8];
    __shared__ double sdb[6];

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
        
        sda[6] = 0;
        
        sda[7] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
        
        sdb[4] = 0;
        
        sdb[5] = 0;
            }

    __syncthreads();
    
    scalar_t d_a0 = 0;
    scalar_t a_0 = a[0];
    
    scalar_t d_a1 = 0;
    scalar_t a_1 = a[1];
    
    scalar_t d_a2 = 0;
    scalar_t a_2 = a[2];
    
    scalar_t d_a3 = 0;
    scalar_t a_3 = a[3];
    
    scalar_t d_a4 = 0;
    scalar_t a_4 = a[4];
    
    scalar_t d_a5 = 0;
    scalar_t a_5 = a[5];
    
    scalar_t d_a6 = 0;
    scalar_t a_6 = a[6];
    
    scalar_t d_a7 = 0;
    scalar_t a_7 = a[7];
    
    
    scalar_t d_b0 = 0;
    scalar_t b_0 = b[0];
    
    scalar_t d_b1 = 0;
    scalar_t b_1 = b[1];
    
    scalar_t d_b2 = 0;
    scalar_t b_2 = b[2];
    
    scalar_t d_b3 = 0;
    scalar_t b_3 = b[3];
    
    scalar_t d_b4 = 0;
    scalar_t b_4 = b[4];
    
    scalar_t d_b5 = 0;
    scalar_t b_5 = b[5];
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {
        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
                scalar_t xp6 = xp5 * xp1;
                scalar_t xp7 = xp6 * xp1;
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
        
        + a_6*xp6
        
        + a_7*xp7
                ;

        scalar_t A =
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                + b_4 * xp5
                + b_5 * xp6
                ;

        scalar_t Q = scalar_t(1.0) + abs(A);

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                + scalar_t(6.0) * a_6 * xp5
                + scalar_t(7.0) * a_7 * xp6
                ;

        scalar_t S = copysign( scalar_t(1.0), A ) * (b_0

                + scalar_t(2.0) * b_1 * xp1
                + scalar_t(3.0) * b_2 * xp2
                + scalar_t(4.0) * b_3 * xp3
                + scalar_t(5.0) * b_4 * xp4
                + scalar_t(6.0) * b_5 * xp5
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

                scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), A ) * xp1;
        d_b0 += d_i_b0 * grad_o;
                scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), A ) * xp2;
        d_b1 += d_i_b1 * grad_o;
                scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), A ) * xp3;
        d_b2 += d_i_b2 * grad_o;
                scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), A ) * xp4;
        d_b3 += d_i_b3 * grad_o;
                scalar_t d_i_b4 = mpq2 * copysign( scalar_t(1.0), A ) * xp5;
        d_b4 += d_i_b4 * grad_o;
                scalar_t d_i_b5 = mpq2 * copysign( scalar_t(1.0), A ) * xp6;
        d_b5 += d_i_b5 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

        
        scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
        
        scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
        
        scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
        
        scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
        
        scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
        
        scalar_t d_i_a6  = xp6/Q;
        d_a6 += d_i_a6 * grad_o;
        
        scalar_t d_i_a7  = xp7/Q;
        d_a7 += d_i_a7 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
    
    atomicAdd(&sda[6], d_a6);
    
    atomicAdd(&sda[7], d_a7);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    atomicAdd(&sdb[4], d_b4);
    
    atomicAdd(&sdb[5], d_b5);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
        
        atomicAdd(&d_a[6], sda[6]);
        
        atomicAdd(&d_a[7], sda[7]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
        
        atomicAdd(&d_b[4], sdb[4]);
        
        atomicAdd(&d_b[5], sdb[5]);
            }
}




std::vector<torch::Tensor> rational_cuda_backward_B_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_B_7_6", ([&] {
    rational_cuda_backward_B_kernel_7_6<scalar_t>
        <<<16, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}




// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / eps + |b_0 + b_1*X + b_2*X^2 + ... + b_n*X^n|
// eps = 0.1


template <typename scalar_t>
__global__ void rational_cuda_forward_C_kernel_5_4( const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    
    scalar_t a_0 = a[0];
    
    scalar_t a_1 = a[1];
    
    scalar_t a_2 = a[2];
    
    scalar_t a_3 = a[3];
    
    scalar_t a_4 = a[4];
    
    scalar_t a_5 = a[5];
    
    
    scalar_t b_0 = b[0];
    
    scalar_t b_1 = b[1];
    
    scalar_t b_2 = b[2];
    
    scalar_t b_3 = b[3];
    
    scalar_t b_4 = b[4];
    
    scalar_t eps = scalar_t(0.1);

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
                ;

        scalar_t Q = eps + abs(b_0
        
        + b_1 * xp1
        
        + b_2 * xp2
        
        + b_3 * xp3
        
        + b_4 * xp4
                );

        result[index] = P/Q;
    }
}


at::Tensor rational_cuda_forward_C_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_C_5_4", ([&] {
    rational_cuda_forward_C_kernel_5_4<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));

    return result;
}

template <typename scalar_t>
__global__ void rational_cuda_backward_C_kernel_5_4(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[6];
    __shared__ double sdb[5];

    scalar_t eps = scalar_t(0.1);

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
        
        sdb[4] = 0;
            }

    __syncthreads();
    
    scalar_t d_a0 = 0;
    scalar_t a_0 = a[0];
    
    scalar_t d_a1 = 0;
    scalar_t a_1 = a[1];
    
    scalar_t d_a2 = 0;
    scalar_t a_2 = a[2];
    
    scalar_t d_a3 = 0;
    scalar_t a_3 = a[3];
    
    scalar_t d_a4 = 0;
    scalar_t a_4 = a[4];
    
    scalar_t d_a5 = 0;
    scalar_t a_5 = a[5];
    
    
    scalar_t d_b0 = 0;
    scalar_t b_0 = b[0];
    
    scalar_t d_b1 = 0;
    scalar_t b_1 = b[1];
    
    scalar_t d_b2 = 0;
    scalar_t b_2 = b[2];
    
    scalar_t d_b3 = 0;
    scalar_t b_3 = b[3];
    
    scalar_t d_b4 = 0;
    scalar_t b_4 = b[4];
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {
        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
                ;

        scalar_t A = b_0
        
        + b_1 * xp1
        
        + b_2 * xp2
        
        + b_3 * xp3
        
        + b_4 * xp4
                ;

        scalar_t Q = eps + abs(A);

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                ;

        scalar_t S = copysign( scalar_t(1.0), A ) * (b_1

                + scalar_t(2.0) * b_2 * xp1
                + scalar_t(3.0) * b_3 * xp2
                + scalar_t(4.0) * b_4 * xp3
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

        scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), A );
        d_b0 += d_i_b0 * grad_o;

        
        scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), A ) * xp1;
        d_b1 += d_i_b1 * grad_o;
        
        scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), A ) * xp2;
        d_b2 += d_i_b2 * grad_o;
        
        scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), A ) * xp3;
        d_b3 += d_i_b3 * grad_o;
        
        scalar_t d_i_b4 = mpq2 * copysign( scalar_t(1.0), A ) * xp4;
        d_b4 += d_i_b4 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

                scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
                scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
                scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
                scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
                scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    atomicAdd(&sdb[4], d_b4);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
        
        atomicAdd(&d_b[4], sdb[4]);
            }
}




std::vector<torch::Tensor> rational_cuda_backward_C_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_C_5_4", ([&] {
    rational_cuda_backward_C_kernel_5_4<scalar_t>
        <<<16, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}



// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / eps + |b_0 + b_1*X + b_2*X^2 + ... + b_n*X^n|
// eps = 0.1


template <typename scalar_t>
__global__ void rational_cuda_forward_C_kernel_7_6( const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    
    scalar_t a_0 = a[0];
    
    scalar_t a_1 = a[1];
    
    scalar_t a_2 = a[2];
    
    scalar_t a_3 = a[3];
    
    scalar_t a_4 = a[4];
    
    scalar_t a_5 = a[5];
    
    scalar_t a_6 = a[6];
    
    scalar_t a_7 = a[7];
    
    
    scalar_t b_0 = b[0];
    
    scalar_t b_1 = b[1];
    
    scalar_t b_2 = b[2];
    
    scalar_t b_3 = b[3];
    
    scalar_t b_4 = b[4];
    
    scalar_t b_5 = b[5];
    
    scalar_t b_6 = b[6];
    
    scalar_t eps = scalar_t(0.1);

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
                scalar_t xp6 = xp5 * xp1;
                scalar_t xp7 = xp6 * xp1;
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
        
        + a_6 * xp6
        
        + a_7 * xp7
                ;

        scalar_t Q = eps + abs(b_0
        
        + b_1 * xp1
        
        + b_2 * xp2
        
        + b_3 * xp3
        
        + b_4 * xp4
        
        + b_5 * xp5
        
        + b_6 * xp6
                );

        result[index] = P/Q;
    }
}


at::Tensor rational_cuda_forward_C_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_C_7_6", ([&] {
    rational_cuda_forward_C_kernel_7_6<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));

    return result;
}

template <typename scalar_t>
__global__ void rational_cuda_backward_C_kernel_7_6(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[8];
    __shared__ double sdb[7];

    scalar_t eps = scalar_t(0.1);

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
        
        sda[6] = 0;
        
        sda[7] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
        
        sdb[4] = 0;
        
        sdb[5] = 0;
        
        sdb[6] = 0;
            }

    __syncthreads();
    
    scalar_t d_a0 = 0;
    scalar_t a_0 = a[0];
    
    scalar_t d_a1 = 0;
    scalar_t a_1 = a[1];
    
    scalar_t d_a2 = 0;
    scalar_t a_2 = a[2];
    
    scalar_t d_a3 = 0;
    scalar_t a_3 = a[3];
    
    scalar_t d_a4 = 0;
    scalar_t a_4 = a[4];
    
    scalar_t d_a5 = 0;
    scalar_t a_5 = a[5];
    
    scalar_t d_a6 = 0;
    scalar_t a_6 = a[6];
    
    scalar_t d_a7 = 0;
    scalar_t a_7 = a[7];
    
    
    scalar_t d_b0 = 0;
    scalar_t b_0 = b[0];
    
    scalar_t d_b1 = 0;
    scalar_t b_1 = b[1];
    
    scalar_t d_b2 = 0;
    scalar_t b_2 = b[2];
    
    scalar_t d_b3 = 0;
    scalar_t b_3 = b[3];
    
    scalar_t d_b4 = 0;
    scalar_t b_4 = b[4];
    
    scalar_t d_b5 = 0;
    scalar_t b_5 = b[5];
    
    scalar_t d_b6 = 0;
    scalar_t b_6 = b[6];
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {
        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
                scalar_t xp6 = xp5 * xp1;
                scalar_t xp7 = xp6 * xp1;
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
        
        + a_6*xp6
        
        + a_7*xp7
                ;

        scalar_t A = b_0
        
        + b_1 * xp1
        
        + b_2 * xp2
        
        + b_3 * xp3
        
        + b_4 * xp4
        
        + b_5 * xp5
        
        + b_6 * xp6
                ;

        scalar_t Q = eps + abs(A);

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                + scalar_t(6.0) * a_6 * xp5
                + scalar_t(7.0) * a_7 * xp6
                ;

        scalar_t S = copysign( scalar_t(1.0), A ) * (b_1

                + scalar_t(2.0) * b_2 * xp1
                + scalar_t(3.0) * b_3 * xp2
                + scalar_t(4.0) * b_4 * xp3
                + scalar_t(5.0) * b_5 * xp4
                + scalar_t(6.0) * b_6 * xp5
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

        scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), A );
        d_b0 += d_i_b0 * grad_o;

        
        scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), A ) * xp1;
        d_b1 += d_i_b1 * grad_o;
        
        scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), A ) * xp2;
        d_b2 += d_i_b2 * grad_o;
        
        scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), A ) * xp3;
        d_b3 += d_i_b3 * grad_o;
        
        scalar_t d_i_b4 = mpq2 * copysign( scalar_t(1.0), A ) * xp4;
        d_b4 += d_i_b4 * grad_o;
        
        scalar_t d_i_b5 = mpq2 * copysign( scalar_t(1.0), A ) * xp5;
        d_b5 += d_i_b5 * grad_o;
        
        scalar_t d_i_b6 = mpq2 * copysign( scalar_t(1.0), A ) * xp6;
        d_b6 += d_i_b6 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

                scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
                scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
                scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
                scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
                scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
                scalar_t d_i_a6  = xp6/Q;
        d_a6 += d_i_a6 * grad_o;
                scalar_t d_i_a7  = xp7/Q;
        d_a7 += d_i_a7 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
    
    atomicAdd(&sda[6], d_a6);
    
    atomicAdd(&sda[7], d_a7);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    atomicAdd(&sdb[4], d_b4);
    
    atomicAdd(&sdb[5], d_b5);
    
    atomicAdd(&sdb[6], d_b6);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
        
        atomicAdd(&d_a[6], sda[6]);
        
        atomicAdd(&d_a[7], sda[7]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
        
        atomicAdd(&d_b[4], sdb[4]);
        
        atomicAdd(&d_b[5], sdb[5]);
        
        atomicAdd(&d_b[6], sdb[6]);
            }
}




std::vector<torch::Tensor> rational_cuda_backward_C_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_C_7_6", ([&] {
    rational_cuda_backward_C_kernel_7_6<scalar_t>
        <<<16, blockSize>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}



// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|



template <typename scalar_t>
__global__ void rational_cuda_forward_D_kernel_5_4(const bool training, const unsigned long long iteration, const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    scalar_t lower = 0;
    scalar_t upper = 0;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        
        scalar_t a_0 = a[0];
        
        scalar_t a_1 = a[1];
        
        scalar_t a_2 = a[2];
        
        scalar_t a_3 = a[3];
        
        scalar_t a_4 = a[4];
        
        scalar_t a_5 = a[5];
        
        
        scalar_t b_0 = b[0];
        
        scalar_t b_1 = b[1];
        
        scalar_t b_2 = b[2];
        
        scalar_t b_3 = b[3];
        
        if(training){
            curandStatePhilox4_32_10_t state;
            curand_init(17, index, iteration*10, &state);

            
            lower = scalar_t(1.0-0.1)*a_0;
            upper = scalar_t(1.0+0.1)*a_0;
            a_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_1;
            upper = scalar_t(1.0+0.1)*a_1;
            a_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_2;
            upper = scalar_t(1.0+0.1)*a_2;
            a_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_3;
            upper = scalar_t(1.0+0.1)*a_3;
            a_3 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_4;
            upper = scalar_t(1.0+0.1)*a_4;
            a_4 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_5;
            upper = scalar_t(1.0+0.1)*a_5;
            a_5 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            
            lower = scalar_t(1.0-0.1)*b_0;
            upper = scalar_t(1.0+0.1)*b_0;
            b_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_1;
            upper = scalar_t(1.0+0.1)*b_1;
            b_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_2;
            upper = scalar_t(1.0+0.1)*b_2;
            b_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_3;
            upper = scalar_t(1.0+0.1)*b_3;
            b_3 = curand_uniform4(&state).x * (upper-lower) + lower;
                    }

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
                ;

        scalar_t Q = scalar_t(1.0) + abs(
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                );

        result[index] = P/Q;
    }
}


at::Tensor rational_cuda_forward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_D_5_4", ([&] {
    rational_cuda_forward_D_kernel_5_4<scalar_t>
        <<<numBlocks, blockSize>>>(
            training, iteration,
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));

    return result;
}




//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |A(X)|
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//A(X) = b_0*X + b_1*X^2 + b_2*X^3
//S(X) = sign(A(X)) * ( b_0 + 2*b_1*X + 3*b_2*X^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,5}
//dF/db_i = (-P(X)/Q(X)^2) * sign(A(X)) * X^{i+1} , i \in {0,4}


template <typename scalar_t>
__global__ void rational_cuda_backward_D_kernel_5_4(
    const bool training, const unsigned long long iteration,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[6];
    __shared__ double sdb[4];

    scalar_t lower = 0;
    scalar_t upper = 0;

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
            }

    __syncthreads();
    
    double d_a0 = 0;
    
    double d_a1 = 0;
    
    double d_a2 = 0;
    
    double d_a3 = 0;
    
    double d_a4 = 0;
    
    double d_a5 = 0;
        
    double d_b0 = 0;
    
    double d_b1 = 0;
    
    double d_b2 = 0;
    
    double d_b3 = 0;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        
        scalar_t a_0 = a[0];
        
        scalar_t a_1 = a[1];
        
        scalar_t a_2 = a[2];
        
        scalar_t a_3 = a[3];
        
        scalar_t a_4 = a[4];
        
        scalar_t a_5 = a[5];
        
        
        scalar_t b_0 = b[0];
        
        scalar_t b_1 = b[1];
        
        scalar_t b_2 = b[2];
        
        scalar_t b_3 = b[3];
        
        if(training){
            curandStatePhilox4_32_10_t state;
            curand_init(17, index, iteration*10, &state);

            
            lower = scalar_t(1.0-0.1)*a_0;
            upper = scalar_t(1.0+0.1)*a_0;
            a_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_1;
            upper = scalar_t(1.0+0.1)*a_1;
            a_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_2;
            upper = scalar_t(1.0+0.1)*a_2;
            a_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_3;
            upper = scalar_t(1.0+0.1)*a_3;
            a_3 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_4;
            upper = scalar_t(1.0+0.1)*a_4;
            a_4 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_5;
            upper = scalar_t(1.0+0.1)*a_5;
            a_5 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            
            lower = scalar_t(1.0-0.1)*b_0;
            upper = scalar_t(1.0+0.1)*b_0;
            b_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_1;
            upper = scalar_t(1.0+0.1)*b_1;
            b_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_2;
            upper = scalar_t(1.0+0.1)*b_2;
            b_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_3;
            upper = scalar_t(1.0+0.1)*b_3;
            b_3 = curand_uniform4(&state).x * (upper-lower) + lower;
                    }

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
                ;

        scalar_t A =
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                ;

        scalar_t Q = scalar_t(1.0) + abs(A);

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                ;

        scalar_t S = copysign( scalar_t(1.0), A ) * (b_0

                + scalar_t(2.0) * b_1 * xp1
                + scalar_t(3.0) * b_2 * xp2
                + scalar_t(4.0) * b_3 * xp3
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

                scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), A ) * xp1;
        d_b0 += d_i_b0 * grad_o;
                scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), A ) * xp2;
        d_b1 += d_i_b1 * grad_o;
                scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), A ) * xp3;
        d_b2 += d_i_b2 * grad_o;
                scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), A ) * xp4;
        d_b3 += d_i_b3 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

        
        scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
        
        scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
        
        scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
        
        scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
        
        scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
            }
}




std::vector<torch::Tensor> rational_cuda_backward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_D_5_4", ([&] {
    rational_cuda_backward_D_kernel_5_4<scalar_t>
        <<<16, blockSize>>>(
            training, iteration,
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}




// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|



template <typename scalar_t>
__global__ void rational_cuda_forward_D_kernel_7_6(const bool training, const unsigned long long iteration, const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    scalar_t lower = 0;
    scalar_t upper = 0;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        
        scalar_t a_0 = a[0];
        
        scalar_t a_1 = a[1];
        
        scalar_t a_2 = a[2];
        
        scalar_t a_3 = a[3];
        
        scalar_t a_4 = a[4];
        
        scalar_t a_5 = a[5];
        
        scalar_t a_6 = a[6];
        
        scalar_t a_7 = a[7];
        
        
        scalar_t b_0 = b[0];
        
        scalar_t b_1 = b[1];
        
        scalar_t b_2 = b[2];
        
        scalar_t b_3 = b[3];
        
        scalar_t b_4 = b[4];
        
        scalar_t b_5 = b[5];
        
        if(training){
            curandStatePhilox4_32_10_t state;
            curand_init(17, index, iteration*14, &state);

            
            lower = scalar_t(1.0-0.1)*a_0;
            upper = scalar_t(1.0+0.1)*a_0;
            a_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_1;
            upper = scalar_t(1.0+0.1)*a_1;
            a_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_2;
            upper = scalar_t(1.0+0.1)*a_2;
            a_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_3;
            upper = scalar_t(1.0+0.1)*a_3;
            a_3 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_4;
            upper = scalar_t(1.0+0.1)*a_4;
            a_4 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_5;
            upper = scalar_t(1.0+0.1)*a_5;
            a_5 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_6;
            upper = scalar_t(1.0+0.1)*a_6;
            a_6 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_7;
            upper = scalar_t(1.0+0.1)*a_7;
            a_7 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            
            lower = scalar_t(1.0-0.1)*b_0;
            upper = scalar_t(1.0+0.1)*b_0;
            b_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_1;
            upper = scalar_t(1.0+0.1)*b_1;
            b_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_2;
            upper = scalar_t(1.0+0.1)*b_2;
            b_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_3;
            upper = scalar_t(1.0+0.1)*b_3;
            b_3 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_4;
            upper = scalar_t(1.0+0.1)*b_4;
            b_4 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_5;
            upper = scalar_t(1.0+0.1)*b_5;
            b_5 = curand_uniform4(&state).x * (upper-lower) + lower;
                    }

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
                scalar_t xp6 = xp5 * xp1;
                scalar_t xp7 = xp6 * xp1;
        
        scalar_t P = a_0
        
        + a_1 * xp1
        
        + a_2 * xp2
        
        + a_3 * xp3
        
        + a_4 * xp4
        
        + a_5 * xp5
        
        + a_6 * xp6
        
        + a_7 * xp7
                ;

        scalar_t Q = scalar_t(1.0) + abs(
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                + b_4 * xp5
                + b_5 * xp6
                );

        result[index] = P/Q;
    }
}


at::Tensor rational_cuda_forward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_D_7_6", ([&] {
    rational_cuda_forward_D_kernel_7_6<scalar_t>
        <<<numBlocks, blockSize>>>(
            training, iteration,
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            x_size);
        }));

    return result;
}




//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |A(X)|
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//A(X) = b_0*X + b_1*X^2 + b_2*X^3
//S(X) = sign(A(X)) * ( b_0 + 2*b_1*X + 3*b_2*X^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,7}
//dF/db_i = (-P(X)/Q(X)^2) * sign(A(X)) * X^{i+1} , i \in {0,6}


template <typename scalar_t>
__global__ void rational_cuda_backward_D_kernel_7_6(
    const bool training, const unsigned long long iteration,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[8];
    __shared__ double sdb[6];

    scalar_t lower = 0;
    scalar_t upper = 0;

    if( threadIdx.x == 0){
        
        sda[0] = 0;
        
        sda[1] = 0;
        
        sda[2] = 0;
        
        sda[3] = 0;
        
        sda[4] = 0;
        
        sda[5] = 0;
        
        sda[6] = 0;
        
        sda[7] = 0;
                
        sdb[0] = 0;
        
        sdb[1] = 0;
        
        sdb[2] = 0;
        
        sdb[3] = 0;
        
        sdb[4] = 0;
        
        sdb[5] = 0;
            }

    __syncthreads();
    
    double d_a0 = 0;
    
    double d_a1 = 0;
    
    double d_a2 = 0;
    
    double d_a3 = 0;
    
    double d_a4 = 0;
    
    double d_a5 = 0;
    
    double d_a6 = 0;
    
    double d_a7 = 0;
        
    double d_b0 = 0;
    
    double d_b1 = 0;
    
    double d_b2 = 0;
    
    double d_b3 = 0;
    
    double d_b4 = 0;
    
    double d_b5 = 0;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        
        scalar_t a_0 = a[0];
        
        scalar_t a_1 = a[1];
        
        scalar_t a_2 = a[2];
        
        scalar_t a_3 = a[3];
        
        scalar_t a_4 = a[4];
        
        scalar_t a_5 = a[5];
        
        scalar_t a_6 = a[6];
        
        scalar_t a_7 = a[7];
        
        
        scalar_t b_0 = b[0];
        
        scalar_t b_1 = b[1];
        
        scalar_t b_2 = b[2];
        
        scalar_t b_3 = b[3];
        
        scalar_t b_4 = b[4];
        
        scalar_t b_5 = b[5];
        
        if(training){
            curandStatePhilox4_32_10_t state;
            curand_init(17, index, iteration*14, &state);

            
            lower = scalar_t(1.0-0.1)*a_0;
            upper = scalar_t(1.0+0.1)*a_0;
            a_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_1;
            upper = scalar_t(1.0+0.1)*a_1;
            a_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_2;
            upper = scalar_t(1.0+0.1)*a_2;
            a_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_3;
            upper = scalar_t(1.0+0.1)*a_3;
            a_3 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_4;
            upper = scalar_t(1.0+0.1)*a_4;
            a_4 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_5;
            upper = scalar_t(1.0+0.1)*a_5;
            a_5 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_6;
            upper = scalar_t(1.0+0.1)*a_6;
            a_6 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*a_7;
            upper = scalar_t(1.0+0.1)*a_7;
            a_7 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            
            lower = scalar_t(1.0-0.1)*b_0;
            upper = scalar_t(1.0+0.1)*b_0;
            b_0 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_1;
            upper = scalar_t(1.0+0.1)*b_1;
            b_1 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_2;
            upper = scalar_t(1.0+0.1)*b_2;
            b_2 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_3;
            upper = scalar_t(1.0+0.1)*b_3;
            b_3 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_4;
            upper = scalar_t(1.0+0.1)*b_4;
            b_4 = curand_uniform4(&state).x * (upper-lower) + lower;
            
            lower = scalar_t(1.0-0.1)*b_5;
            upper = scalar_t(1.0+0.1)*b_5;
            b_5 = curand_uniform4(&state).x * (upper-lower) + lower;
                    }

        scalar_t xp1 = x[index];

                scalar_t xp2 = xp1 * xp1;
                scalar_t xp3 = xp2 * xp1;
                scalar_t xp4 = xp3 * xp1;
                scalar_t xp5 = xp4 * xp1;
                scalar_t xp6 = xp5 * xp1;
                scalar_t xp7 = xp6 * xp1;
        
        scalar_t P = a_0
        
        + a_1*xp1
        
        + a_2*xp2
        
        + a_3*xp3
        
        + a_4*xp4
        
        + a_5*xp5
        
        + a_6*xp6
        
        + a_7*xp7
                ;

        scalar_t A =
                + b_0 * xp1
                + b_1 * xp2
                + b_2 * xp3
                + b_3 * xp4
                + b_4 * xp5
                + b_5 * xp6
                ;

        scalar_t Q = scalar_t(1.0) + abs(A);

        scalar_t R = a_1
                + scalar_t(2.0) * a_2 * xp1
                + scalar_t(3.0) * a_3 * xp2
                + scalar_t(4.0) * a_4 * xp3
                + scalar_t(5.0) * a_5 * xp4
                + scalar_t(6.0) * a_6 * xp5
                + scalar_t(7.0) * a_7 * xp6
                ;

        scalar_t S = copysign( scalar_t(1.0), A ) * (b_0

                + scalar_t(2.0) * b_1 * xp1
                + scalar_t(3.0) * b_2 * xp2
                + scalar_t(4.0) * b_3 * xp3
                + scalar_t(5.0) * b_4 * xp4
                + scalar_t(6.0) * b_5 * xp5
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

                scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), A ) * xp1;
        d_b0 += d_i_b0 * grad_o;
                scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), A ) * xp2;
        d_b1 += d_i_b1 * grad_o;
                scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), A ) * xp3;
        d_b2 += d_i_b2 * grad_o;
                scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), A ) * xp4;
        d_b3 += d_i_b3 * grad_o;
                scalar_t d_i_b4 = mpq2 * copysign( scalar_t(1.0), A ) * xp5;
        d_b4 += d_i_b4 * grad_o;
                scalar_t d_i_b5 = mpq2 * copysign( scalar_t(1.0), A ) * xp6;
        d_b5 += d_i_b5 * grad_o;
        
        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

        
        scalar_t d_i_a1  = xp1/Q;
        d_a1 += d_i_a1 * grad_o;
        
        scalar_t d_i_a2  = xp2/Q;
        d_a2 += d_i_a2 * grad_o;
        
        scalar_t d_i_a3  = xp3/Q;
        d_a3 += d_i_a3 * grad_o;
        
        scalar_t d_i_a4  = xp4/Q;
        d_a4 += d_i_a4 * grad_o;
        
        scalar_t d_i_a5  = xp5/Q;
        d_a5 += d_i_a5 * grad_o;
        
        scalar_t d_i_a6  = xp6/Q;
        d_a6 += d_i_a6 * grad_o;
        
        scalar_t d_i_a7  = xp7/Q;
        d_a7 += d_i_a7 * grad_o;
            }

    
    atomicAdd(&sda[0], d_a0);
    
    atomicAdd(&sda[1], d_a1);
    
    atomicAdd(&sda[2], d_a2);
    
    atomicAdd(&sda[3], d_a3);
    
    atomicAdd(&sda[4], d_a4);
    
    atomicAdd(&sda[5], d_a5);
    
    atomicAdd(&sda[6], d_a6);
    
    atomicAdd(&sda[7], d_a7);
        
    atomicAdd(&sdb[0], d_b0);
    
    atomicAdd(&sdb[1], d_b1);
    
    atomicAdd(&sdb[2], d_b2);
    
    atomicAdd(&sdb[3], d_b3);
    
    atomicAdd(&sdb[4], d_b4);
    
    atomicAdd(&sdb[5], d_b5);
    
    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_a[0], sda[0]);
        
        atomicAdd(&d_a[1], sda[1]);
        
        atomicAdd(&d_a[2], sda[2]);
        
        atomicAdd(&d_a[3], sda[3]);
        
        atomicAdd(&d_a[4], sda[4]);
        
        atomicAdd(&d_a[5], sda[5]);
        
        atomicAdd(&d_a[6], sda[6]);
        
        atomicAdd(&d_a[7], sda[7]);
                
        atomicAdd(&d_b[0], sdb[0]);
        
        atomicAdd(&d_b[1], sdb[1]);
        
        atomicAdd(&d_b[2], sdb[2]);
        
        atomicAdd(&d_b[3], sdb[3]);
        
        atomicAdd(&d_b[4], sdb[4]);
        
        atomicAdd(&d_b[5], sdb[5]);
            }
}




std::vector<torch::Tensor> rational_cuda_backward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_D_7_6", ([&] {
    rational_cuda_backward_D_kernel_7_6<scalar_t>
        <<<16, blockSize>>>(
            training, iteration,
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<double>(),
            d_d.data_ptr<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}



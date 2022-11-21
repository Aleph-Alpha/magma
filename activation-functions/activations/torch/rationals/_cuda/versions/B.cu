#foreach( $degs in $degrees )

// P(X)/Q(X) = a_0 + a_1*X + a_2*X^2 + ... + a_n*X^n / 1 + |b_1*X + b_2*X^2 + ... + b_n*X^n|


#set( $degs_a = $degs[0] )
#set( $degs_b = $degs[1] )
#set( $coefs_a = $degs_a )
#set( $coefs_b = $degs_b - 1 )
#set( $a_counts = $coefs_a + 1 )
#set( $b_counts = $coefs_b + 1 )
#set( $max_x = $degs[2] )

template <typename scalar_t>
__global__ void rational_cuda_forward_B_kernel_$degs[0]_$degs[1]( const scalar_t* __restrict__ x, const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, scalar_t* __restrict__ result, size_t x_size) {

    #foreach( $idx in [0..$coefs_a] )
    scalar_t a_$idx = a[$idx];
    #end

    #foreach( $idx in [0..$coefs_b] )
    scalar_t b_$idx = b[$idx];
    #end

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];

        #foreach( $idx in [2..$max_x] )#set( $value = $idx - 1 )
        scalar_t xp$idx = xp$value * xp1;
        #end

        scalar_t P = a_0
        #foreach( $idx in [1..$coefs_a] )
        + a_$idx * xp$idx
        #end
        ;

        scalar_t Q = scalar_t(1.0) + abs(
        #foreach( $idx in [0..$coefs_b] )#set( $value = $idx + 1 )
        + b_$idx * xp$value
        #end
        );

        result[index] = P/Q;
    }
}


at::Tensor rational_cuda_forward_B_$degs[0]_$degs[1](torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_forward_B_$degs[0]_$degs[1]", ([&] {
    rational_cuda_forward_B_kernel_$degs[0]_$degs[1]<scalar_t>
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
//dF/da_i = x^i/Q(X), i \in {0,$degs[0]}
//dF/db_i = (-P(X)/Q(X)^2) * sign(A(X)) * X^{i+1} , i \in {0,$degs[1]}


template <typename scalar_t>
__global__ void rational_cuda_backward_B_kernel_$degs[0]_$degs[1](
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_a,
    double* __restrict__ d_b,
    size_t x_size) {

    __shared__ double sda[$a_counts];
    __shared__ double sdb[$b_counts];

    if( threadIdx.x == 0){
        #foreach( $idx in [0..$coefs_a] )
        sda[$idx] = 0;
        #end
        #foreach( $idx in [0..$coefs_b] )
        sdb[$idx] = 0;
        #end
    }

    __syncthreads();
    #foreach( $idx in [0..$coefs_a] )
    scalar_t d_a$idx = 0;
    scalar_t a_$idx = a[$idx];
    #end

    #foreach( $idx in [0..$coefs_b] )
    scalar_t d_b$idx = 0;
    scalar_t b_$idx = b[$idx];
    #end

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {
        scalar_t xp1 = x[index];

        #foreach( $idx in [2..$max_x] )#set( $value = $idx - 1 )
        scalar_t xp$idx = xp$value * xp1;
        #end

        scalar_t P = a_0
        #foreach( $idx in [1..$coefs_a] )
        + a_$idx*xp$idx
        #end
        ;

        scalar_t A =
        #foreach( $idx in [0..$coefs_b] )#set( $value = $idx + 1 )
        + b_$idx * xp$value
        #end
        ;

        scalar_t Q = scalar_t(1.0) + abs(A);

        scalar_t R = a_1
        #foreach( $idx in [2..$coefs_a] )#set( $value = $idx - 1 )
        + scalar_t($idx.0) * a_$idx * xp$value
        #end
        ;

        scalar_t S = copysign( scalar_t(1.0), A ) * (b_0

        #foreach( $idx in [1..$coefs_b] )#set( $value = $idx + 1 )
        + scalar_t($value.0) * b_$idx * xp$idx
        #end
         );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2);
        d_x[index] = d_i_x * grad_o;

        #foreach( $idx in [0..$coefs_b] )#set( $value = $idx + 1 )
        scalar_t d_i_b$idx = mpq2 * copysign( scalar_t(1.0), A ) * xp$value;
        d_b$idx += d_i_b$idx * grad_o;
        #end

        scalar_t d_i_a0 = scalar_t(1.0)/Q;
        d_a0 += d_i_a0 * grad_o;

        #foreach( $idx in [1..$coefs_a] )
        scalar_t d_i_a$idx  = xp$idx/Q;
        d_a$idx += d_i_a$idx * grad_o;
        #end
    }

    #foreach( $idx in [0..$coefs_a] )
    atomicAdd(&sda[$idx], d_a$idx);
    #end
    #foreach( $idx in [0..$coefs_b] )
    atomicAdd(&sdb[$idx], d_b$idx);
    #end

    __syncthreads();

    if( threadIdx.x == 0){
        #foreach( $idx in [0..$coefs_a] )
        atomicAdd(&d_a[$idx], sda[$idx]);
        #end
        #foreach( $idx in [0..$coefs_b] )
        atomicAdd(&d_b[$idx], sdb[$idx]);
        #end
    }
}




std::vector<torch::Tensor> rational_cuda_backward_B_$degs[0]_$degs[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_cuda_backward_B_$degs[0]_$degs[1]", ([&] {
    rational_cuda_backward_B_kernel_$degs[0]_$degs[1]<scalar_t>
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


#end

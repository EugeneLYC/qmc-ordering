using LinearAlgebra, Statistics
using Distributions, QuasiMonteCarlo, Random

n = 10000
d = 10

# synthetic d-dimensional least squares problem
wgen = rand(d);
w0 = zeros(d);

# Quasi-random distributions
qrngs = Dict([("IID Uniform"     , UniformSample()), 
              ("Sobol"           , SobolSample()),])

qrng_names = collect(keys(qrngs))

# data distribution
# x ~ N(0,I_d)
# y ~ N(x'wopt, 1)
# goal is to learn w, which is to minimize E[0.5*(x'w - y)^2]

# online mode:  draw samples from this integral at every SGD iteration
# offline mode: draw n samples iid uniformly from this integral 
#               to form a training set,
#               then draw one sample (possibly following a low-discrepancy sequence) 
#               from the training set at every SGD iteration

function gen_xy(q::Vector{Float64})
    xi = quantile(Normal(), q[1:d])  
    yi = quantile(Normal(xi'wgen, 1), q[end])
    return (xi, yi)
end


function gen_syn_data(num_examples::Int64, qrng_name::String, dim::Int64, seed::Int64)
    Random.seed!(seed) 
    
    lb = zeros(dim)
    ub = ones(dim)
    
    distrib = qrngs[qrng_name]
    qmc_samples = QuasiMonteCarlo.sample(num_examples, lb, ub, distrib)

    syn_data = []
    for i in 1:num_examples
        q = mod.(qmc_samples[:,i], 1)
        push!(syn_data, gen_xy(q))
        
    end
    return syn_data
end


function sgd(T::Int64, w0::Vector{Float64}, wopt::Vector{Float64},
	samples::Vector{Any}, ordering::Vector{Int64})
	subopt = zeros(T)
	wk = w0
	# diminishing step size optimized for IID Uniform
	alpha = 1/(d + 2 + d/norm(wgen)^2);
	for t in 1:T
		(xi, yi) = samples[ordering[t]]
		alpha = 1/(1/alpha + (1 - alpha*(d+2))/(1 - alpha))
		wk -= alpha * (xi'wk - yi) * xi
		subopt[t] = norm(wk - wopt).^2
	end
	return subopt
end
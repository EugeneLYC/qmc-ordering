using LinearAlgebra, Statistics
using Distributions, QuasiMonteCarlo, Random
using IterTools, ProgressBars
using JLD
using LaTeXStrings
using PyPlot

SEED    = 42
Random.seed!(SEED)

RUNS    = 1
EPOCHS  = 1000
n       = 10000
d       = 10
T       = n * EPOCHS


# synthetic d-dimensional least squares problem
wgen = rand(d);

# Quasi-random distributions
qrngs = Dict([("IID Uniform"     , UniformSample()), 
              ("Sobol"           , SobolSample())])
qrng_names = vcat(collect(keys(qrngs)), ["RR", "SO"])

# data distribution
# x ~ N(0,I_d)
# y ~ N(x'wopt, 1)
# goal is to learn w, which is to minimize E[0.5*(x'w - y)^2]

# online mode:  draw samples from this integral at every SGD iteration
# offline mode: draw n samples iid uniformly from this integral 
#               to form a training set,
#               then draw one sample from the training set 
#               at every SGD iteration
# note here the samples used in SGD are the ones that are drawn
# with a low-discrepency sequence

function gen_xy(q::Vector{Float64})
    xi = quantile(Normal(), q[1:d])  
    yi = quantile(Normal(xi'wgen, 1), q[end])
    return (xi, yi)
end


function gen_syn_data(num_examples::Int64, qrng_name::String, dim::Int64, seed::Int64)
    Random.seed!(SEED) 
    
    lb = zeros(dim)
    ub = ones(dim)
    
    distrib = qrngs[qrng_name]
    qmc_samples = QuasiMonteCarlo.sample(num_examples, lb, ub, distrib)

    syn_data = []
    v = occursin("Randomized", qrng_name) ? rand(dim) : zeros(dim)
    for i in 1:num_examples
        q = mod.(qmc_samples[:,i] + v, 1)
        push!(syn_data, gen_xy(q))
        
    end
    return syn_data
end


function gen_syn_data_all()
    # generate the training set consisting of n examples
    # sampled iid uniformly from the underlying distribution, 
    # and the same training set will be used for all runs,
    gaussian_samples = gen_syn_data(n, "IID Uniform", d+1, SEED)
    return gaussian_samples
end


function grad_errors(T::Int64, wtau::Vector{Float64}, fullgrad::Vector{Float64},
             samples::Vector{Any}, ordering::Vector{Int64})
    errs = zeros(T)
    acc = zeros(d)
    for m in 1:T
        (xi, yi) = samples[ordering[m]]
        acc += (xi'wtau - yi) * xi
        errs[m] = norm(acc/m - fullgrad)^2
    end
    return errs
end


# training
result_fn = "off-graderr-$n-$d-$RUNS-$EPOCHS-$SEED-$qrng_names.jld"
if  isfile(result_fn)
    println("loading results from $result_fn")
    result = load(result_fn)
    println("results loaded")
else
    println("generating data...")
    samples_offline = gen_syn_data_all() 

    println("begin computing gradient errors...")
    result = Dict()

    w0 = ones(d)

    X = hcat([pair[1] for pair in samples_offline]...)'
    y = [pair[2] for pair in samples_offline]
    full_grad = X'*(X*w0-y)/n

    for qrng_name in qrng_names    
        err_list = zeros(RUNS, T)
        
        for run in 1:RUNS      
            if qrng_name in collect(keys(qrngs))
                distrib = qrngs[qrng_name]
                # generate QMC indices 
                sample_indices = [Int(ceil(i)) for i in QuasiMonteCarlo.sample(T, 0, n, distrib)]
            else
                if qrng_name == "SO"
                    Random.seed!(SEED + run)
                    perm = randperm!(collect(1:n))
                    sample_indices = repeat(perm, EPOCHS)

                elseif qrng_name == "RR"
                    sample_indices = zeros(Int64, T)
                    for epoch in 1:EPOCHS
                        Random.seed!(SEED + run + epoch)
                        perm = randperm!(collect(1:n))
                        sample_indices[(epoch-1)*n+1 : epoch*n] = perm
                    end
                else
                    ArgumentError("SGD variant does not exist.")
                end
            end

            err_list[run, :] = grad_errors(T, w0, full_grad, samples_offline, sample_indices)
        end
        result[qrng_name] = err_list
    end

    println("completed")

    save(result_fn, result)
    println("results file saved at $result_fn")
end

println("plotting...")

figure(figsize = (6,6))
colors = ["#1f77bf", "#ff7f0e", "#17becf", "#7f7f7f"]
for (q, qrng_name) in enumerate(["IID Uniform", "Sobol", "RR", "SO"])

    err_list = result[qrng_name]
    err_mean = mean(err_list, dims=1)'

    # use moving average to smooth out the curves as RR and SO are highly oscillatory
    # towards the end
    using RollingFunctions
    ws = 2000
    err_mean = runmean(reshape(err_mean, T), ws)
    
    loglog(collect(1000:1000:T), err_mean[1000:1000:T], label=qrng_name, color=colors[q])
    ylim(bottom=1e-8)
end

legend(loc="lower left", fontsize=18)

xlabel(L"m", fontsize=20)
ylabel(L"\Vert \frac{1}{m}\Sigma_{t=0}^{m-1}\nabla R_n(w_0; x_t) - \nabla R_n(w_0)\Vert^2", fontsize=20)
title("Offline", fontsize=24)
xticks(fontsize=20)
yticks(fontsize=20)

tight_layout()
plot_fn = "qmc-graderr-offline.pdf"
savefig(plot_fn)
println("plot saved at $plot_fn")


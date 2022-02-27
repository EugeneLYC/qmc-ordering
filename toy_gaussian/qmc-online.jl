using LinearAlgebra, Statistics
using Distributions, QuasiMonteCarlo, Random
using IterTools, ProgressBars
using JLD
using LaTeXStrings, PyPlot

SEED    = 42
Random.seed!(SEED)

RUNS    = 1
EPOCHS  = 1000
n       = 10000
d       = 10
T       = n * EPOCHS

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
    gaussian_samples = Dict()
    # for each distribution, pre-generate the stream of examples
    # to be used in T iterations of online SGD
    # for a specified number of runs
    iter_qrngs = ProgressBar(qrngs)
    for (qrng_name, ~) in iter_qrngs
        println(iter_qrngs, "generating data for $qrng_name...")

        data = []
        iter_runs = ProgressBar(1:RUNS)
        for run in iter_runs
            println(iter_runs, "working on run $run")
            data_run = gen_syn_data(T, qrng_name, d+1, SEED+run)
            push!(data, data_run)
        end
        gaussian_samples[qrng_name] = data
    end
    return gaussian_samples
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


# training
result_fn = "on--$n-$d-$RUNS-$EPOCHS-$SEED-$qrng_names.jld"
if isfile(result_fn)
    println("loading results from $result_fn")
    result = load(result_fn)
    println("results loaded")
else
    println("generating data...")
    samples_online = gen_syn_data_all() 

    println("begin training...")
    result = Dict()
    
    for qrng_name in ProgressBar(qrng_names)
        subopt_list = zeros(RUNS, T)
        
        for run in 1:RUNS            
            # run sgd for online mode
            samples = samples_online[qrng_name][run]
            subopt_list[run, :] = sgd(T, w0, wgen, samples, collect(1:T))
        end
        result[qrng_name] = subopt_list
    end
    println("training completed")

    save(result_fn, result)
    println("results file saved at $result_fn")
end


println("plotting...")

# final plotting
figure(figsize = (6,6))
for qrng_name in ["IID Uniform", "Sobol"]
    subopt_list = result[qrng_name]
    subopt_mean = mean(subopt_list, dims=1)'[100:100:T]

    loglog(collect(100:100:T), subopt_mean, label=qrng_name)
end

loglog(collect(100:100:T), d * (collect(100:100:T)).^(-1), label=L"1/t", color="#d62728", linewidth=3, linestyle="dashed")
loglog(collect(100:100:T), 16 * d^2 *(collect(100:100:T)).^(-2), label=L"1/t^2", color="#2ca02c", linewidth=3, linestyle="dashed")
legend(loc="lower left", fontsize=18)

xlabel(string("iterations ", L"t"), fontsize=20)
ylabel(L"\Vert w_t-w^*\Vert^2", fontsize=20)
title("Online", fontsize=24)
xticks(fontsize=20)
yticks(fontsize=20)

ax = gca()
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.25,0.5,0.75), numticks=50)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

tight_layout()
plot_fn = "qmc-online.pdf"
savefig(plot_fn)
println("plot saved at $plot_fn")



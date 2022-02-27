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
result_fn = "off--$n-$d-$RUNS-$EPOCHS-$SEED-$qrng_names.jld"
if isfile(result_fn)
    println("loading results from $result_fn")
    result = load(result_fn)
    println("results loaded")
else
    println("generating data...")
    samples_offline = gen_syn_data_all() 

    # compute wopt, which is different from wgen since the objective is a finite sum
    # so wopt should be the solution to the lstsq problem

    X = hcat([pair[1] for pair in samples_offline]...)'
    y = [pair[2] for pair in samples_offline]

    wopt = (X'X)\(X'y)

    println("begin training...")
    result = Dict()
        
    for qrng_name in qrng_names
        subopt_list = zeros(RUNS, T)
        
        for run in 1:RUNS            
            # for offline mode we first need to determine sample ordering of the training set
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
            
            # run sgd for offline mode
            subopt_list[run, :] = sgd(T, w0, wopt, samples_offline, sample_indices)            
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
colors = ["#1f77bf", "#ff7f0e", "#17becf", "#7f7f7f"]
for (q, qrng_name) in enumerate(["IID Uniform", "Sobol", "RR", "SO"])
    subopt_list = result[qrng_name]
    subopt_mean = mean(subopt_list, dims=1)

    # use moving average to smooth out the curves as RR and SO are highly oscillatory
    # towards the end
    using RollingFunctions
    ws = 2000
    subopt_mean = runmean(reshape(subopt_mean, T), ws)

    loglog(collect(100:100:T), subopt_mean[100:100:T], label=qrng_name, color=colors[q])
end

legend(loc="lower left", fontsize=18)

xlabel(string("iterations ", L"t"), fontsize=20)
ylabel(L"\Vert w_t-w^*_n\Vert^2", fontsize=20)
title("Offline", fontsize=24)
xticks(fontsize=20)
yticks(fontsize=20)

ax = gca()
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.25,0.5,0.75), numticks=50)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


tight_layout()
plot_fn = "qmc-offline.pdf"
savefig(plot_fn)
println("plot saved at $plot_fn")
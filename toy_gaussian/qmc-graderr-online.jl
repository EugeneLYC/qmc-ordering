using IterTools, ProgressBars
using JLD
using LaTeXStrings
using PyPlot

include("synthetic.jl")

SEED    = 42
Random.seed!(SEED)

RUNS    = 1
EPOCHS  = 1000
T       = n * EPOCHS


# synthetic d-dimensional least squares problem
wgen = rand(d);

# Quasi-random distributions
qrngs = Dict([("IID Uniform"     , UniformSample()), 
              ("Sobol"           , SobolSample())])
qrng_names = collect(keys(qrngs))


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


println("generating data...")
samples_online = gen_syn_data_all() 

println("begin computing gradient errors...")
result = Dict()

w0 = zeros(d)
for qrng_name in qrng_names
    err_list = zeros(RUNS, T)
    
    for run in 1:RUNS            
        samples = samples_online[qrng_name][run]
        full_grad = w0 - wgen   
        err_list[run, :] = grad_errors(T, w0, full_grad, samples, collect(1:T))
    end
    result[qrng_name] = err_list
end

println("completed")
println("plotting...")

figure(figsize = (6,6))
for qrng_name in ["IID Uniform", "Sobol"]
    err_list = result[qrng_name]
    err_mean = mean(err_list, dims=1)'[1000:1000:T]

    loglog(collect(1000:1000:T), err_mean, label=qrng_name)
end

legend(loc="lower left", fontsize=18)

xlabel(L"m", fontsize=20)
ylabel(L"\Vert \frac{1}{m}\Sigma_{t=0}^{m-1}\nabla R(w_0; x_t) - \nabla R(w_0)\Vert^2", fontsize=20)
title("Online", fontsize=24)
xticks(fontsize=20)
yticks(fontsize=20)

ax = gca()
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.25,0.5,0.75), numticks=50)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

tight_layout()
plot_fn = joinpath(pwd(),"figs", "qmc-graderr-online.pdf")
savefig(plot_fn)
println("plot saved at $plot_fn")


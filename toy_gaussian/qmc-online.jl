using IterTools, ProgressBars
using JLD
using LaTeXStrings, PyPlot

include("synthetic.jl")

RUNS    = 1
EPOCHS  = 1000
T       = n * EPOCHS

SEED    = 42
Random.seed!(SEED)

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
plot_fn = joinpath(pwd(),"figs", "qmc-online.pdf")
savefig(plot_fn)
println("plot saved at $plot_fn")



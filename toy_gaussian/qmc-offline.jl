using IterTools, ProgressBars
using JLD
using LaTeXStrings, PyPlot

include("synthetic.jl")
qrng_names = vcat(collect(keys(qrngs)), ["RR", "SO"])

RUNS    = 1
EPOCHS  = 1000
T       = n * EPOCHS

SEED    = 42
Random.seed!(SEED)


function gen_syn_data_all()
    # generate the training set consisting of n examples
    # sampled iid uniformly from the underlying distribution, 
    # and the same training set will be used for all runs,
    gaussian_samples = gen_syn_data(n, "IID Uniform", d+1, SEED)
    return gaussian_samples
end


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
plot_fn = joinpath(pwd(),"figs", "qmc-offline.pdf")
savefig(plot_fn)
println("plot saved at $plot_fn")
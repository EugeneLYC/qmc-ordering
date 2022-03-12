using IterTools, ProgressBars
using JLD
using LaTeXStrings
using PyPlot

include("synthetic.jl")
qrng_names = vcat(collect(keys(qrngs)), ["RR", "SO"])

SEED    = 42
Random.seed!(SEED)

RUNS    = 1
EPOCHS  = 1000
T       = n * EPOCHS

# synthetic d-dimensional least squares problem
wgen = rand(d);

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
plot_fn = joinpath(pwd(),"figs", "qmc-graderr-offline.pdf")
savefig(plot_fn)
println("plot saved at $plot_fn")


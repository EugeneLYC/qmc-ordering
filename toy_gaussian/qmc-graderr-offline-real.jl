using LinearAlgebra, Statistics
using Distributions, QuasiMonteCarlo, Random
using IterTools, ProgressBars
using LaTeXStrings
using PyPlot
using NPZ, JLD

SEED    = 42
Random.seed!(SEED)

# binary logistic regression objective with l2 regularization
a6a = npzread("a6a.npy")
y = a6a[:,1]
X = a6a[:,2:end]

RUNS    = 1
EPOCHS  = 1000
n       = size(X)[1]
d       = size(X)[2]
T       = n * EPOCHS
λ       = 1e-4

# Quasi-random distributions
qrngs = Dict([("IID Uniform"     , UniformSample()), 
              ("Sobol"           , SobolSample())])
qrng_names = vcat(collect(keys(qrngs)), ["RR", "SO"])


function grad_errors(T::Int64, wtau::Vector{Float64}, 
                     fullgrad::Vector{Float64}, ordering::Vector{Int64})
    errs = zeros(T)
    acc = zeros(d)
    for m in 1:T
        i = ordering[m]
        xi = X[i,:]
        yi = y[i]
        res = - yi / (1 + exp(yi * xi'wtau))
        g = xi * res + λ * wtau
        acc +=  g
        errs[m] = norm(acc/m - fullgrad)^2
    end
    return errs
end


# training
result_fn = "off-graderr-a6a-$n-$d-$RUNS-$EPOCHS-$SEED-$qrng_names.jld"
if isfile(result_fn)
    println("loading results from $result_fn")
    result = load(result_fn)
    println("results loaded")
else
    println("begin computing gradient errors...")
    result = Dict()

    w0 = zeros(d)
    res = -y ./ (1 .+ exp.(y .* X * w0))
    full_grad = reshape(sum(X .* res, dims = 1) ./ n, d) + λ * w0

    for qrng_name in qrng_names
        err_list = zeros(RUNS, T)

        for run = 1:RUNS
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
                    for epoch = 1:EPOCHS
                        Random.seed!(SEED + run + epoch)
                        perm = randperm!(collect(1:n))
                        sample_indices[(epoch-1)*n+1:epoch*n] = perm
                    end
                else
                    ArgumentError("SGD variant does not exist.")
                end
            end
        
            err_list[run, :] = grad_errors(T, w0, full_grad, sample_indices)
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
    # using RollingFunctions
    # ws = 2000
    # err_mean = runmean(reshape(err_mean, T), ws)
    
    loglog(collect(1000:1000:T), err_mean[1000:1000:T], label=qrng_name, color=colors[q])
    ylim(bottom=1e-8)
end

legend(loc="upper right", fontsize=18)

xlabel(L"m", fontsize=20)
ylabel(L"\Vert \frac{1}{m}\Sigma_{t=0}^{m-1}\nabla R_n(w_0; x_t) - \nabla R_n(w_0)\Vert^2", fontsize=20)
title("Offline (a6a)", fontsize=24)
xticks(fontsize=20)
yticks(fontsize=20)

tight_layout()
plot_fn = "qmc-graderr-offline-a6a.pdf"
savefig(plot_fn)
println("plot saved at $plot_fn")


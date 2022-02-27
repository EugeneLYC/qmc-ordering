using LinearAlgebra, Statistics
using Distributions, QuasiMonteCarlo, Random
using IterTools, ProgressBars
using NPZ, JLD
using LaTeXStrings, PyPlot

SEED = 42
Random.seed!(SEED)

# binary logistic regression objective with l2 regularization
a6a = npzread("a6a.npy")
y = a6a[:, 1]
X = a6a[:, 2:end]

RUNS = 1
EPOCHS = 5000
n = size(X)[1]
d = size(X)[2]
T = n * EPOCHS
λ = 1e-4

# Quasi-random distributions
qrngs = Dict([("IID Uniform", UniformSample()),
    ("Sobol", SobolSample())])

qrng_names = vcat(collect(keys(qrngs)), ["RR", "SO"])

# qrngs = Dict([("IID Uniform", UniformSample())])

# qrng_names = vcat(collect(keys(qrngs)))

function sgd(T::Int64, w0::Vector{Float64}, wopt::Vector{Float64},
    α0::Float64, ordering::Vector{Int64})
    subopt = zeros(T)
    wk = w0

    for t = 1:T
        i = ordering[t]
        xi = X[i, :]
        yi = y[i]

        res = -yi / (1 + exp(yi * xi'wk))
        g = xi * res + λ * wk

        α = 1000. / (t + 1000.)
        wk -= α * g

        subopt[t] = norm(wk - wopt) .^ 2

    end
    return subopt
end

function compute_wopt(w0)
    wk = w0
    α = 1
    for t = 1:1e4
        yXw = y .* (X * wk)
        res = -y ./ (1 .+ exp.(yXw))
        g = reshape(sum(X .* res, dims = 1) ./ n, d) + λ * wk
        g_norm = norm(g)

        if t % 100 == 0
            f = sum(log.(1 .+ exp.(-yXw))) / n + (λ / 2) * (wk'wk)
            println("iter $t: loss $f, grad_norm $g_norm")
        end

        if g_norm < 1e-15
            break
        end

        if t >= 1000
            σ = 1 ./ (1 .+ exp.(-yXw))
            s = σ .* (1 .- σ)
            H = X' * (s .* X) ./ n + λ * I
            # newton update
            wk -= H \ g
        else
            # gradient update
            wk -= α * g
        end
    end
    return wk
end

# training
result_fn = "off--$n-$d-$RUNS-$EPOCHS-$SEED-$qrng_names-a6a.jld"
if isfile(result_fn)
    println("loading results from $result_fn")
    result = load(result_fn)
    println("results loaded")
else
    w0 = zeros(d)

    # compute wopt by running gradient descent for a large number of iters
    wopt_fn = "a6a-logl2-wopt.jld"
    if isfile(wopt_fn)
        println("loading wopt from $wopt_fn")
        wopt = load(wopt_fn)["wopt"]
        println("wopt loaded")
    else
        println("computing wopt on l2 regularized logistic regression for a6a...")
        wopt = compute_wopt(w0)
        save(wopt_fn, Dict("wopt" => wopt))
        println("wopt file saved at $wopt_fn")
    end

    println("begin training...")
    result = Dict()

    for qrng_name in qrng_names
        subopt_list = zeros(RUNS, T)

        for run = 1:RUNS
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
                    for epoch = 1:EPOCHS
                        Random.seed!(SEED + run + epoch)
                        perm = randperm!(collect(1:n))
                        sample_indices[(epoch-1)*n+1:epoch*n] = perm
                    end
                else
                    ArgumentError("SGD variant does not exist.")
                end
            end

            # run sgd for offline mode
            subopt_list[run, :] = sgd(T, w0, wopt, 1., sample_indices)
        end
        result[qrng_name] = subopt_list
    end

    println("training completed")

    save(result_fn, result)
    println("results file saved at $result_fn")
end

println("plotting...")

# final plotting
figure(figsize = (6, 6))
colors = ["#1f77bf", "#ff7f0e", "#17becf", "#7f7f7f"]
for (q, qrng_name) in enumerate(["IID Uniform", "Sobol", "RR", "SO"])
    subopt_list = result[qrng_name]
    subopt_mean = mean(subopt_list, dims = 1)
    # use moving average to smooth out the curves as RR and SO are highly oscillatory
    # towards the end
    # using RollingFunctions
    # ws = 2000
    # subopt_mean = runmean(reshape(subopt_mean, T), ws)

    loglog(collect(100:100:T), subopt_mean[100:100:T], label = qrng_name, color = colors[q])
end

legend(loc = "upper right", fontsize = 18)

xlabel(string("iterations ", L"t"), fontsize = 20)
ylabel(L"\Vert w_t-w^*_n\Vert^2", fontsize = 20)
title("Offline (a6a)", fontsize = 24)
xticks(fontsize = 20)
yticks(fontsize = 20)

ax = gca()
locmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.25, 0.5, 0.75), numticks = 50)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


tight_layout()
plot_fn = "qmc-offline-a6a.pdf"
savefig(plot_fn)
println("plot saved at $plot_fn")
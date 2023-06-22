using LinearAlgebra
using Random

using JLD2

using LaTeXStrings
using Plots; pgfplotsx()
default(framestyle = :box)

using Logging, LoggingExtras

include("./algorithms.jl")
include("./problems.jl")

#===================================================
LASSO problems
===================================================#

function test_lasso_rmd(x★::Vector{Float64}, A::Matrix{Float64}, n::Int64, r::Float64; K::Int64 = 20, max_iters::Union{Nothing,Int64}=nothing)
    A_norm = 2 # this holds w.h.p
    prob = LassoProblem(A, A * x★, r,  norm(x★,1), 1 + r * A_norm, n, 1 + log(n))

    rmd_iter_info = Iterate_info(:LASSO, x★, 1000, max_iters)
    polyak_rmd(prob, zeros(n), K; iter_info = rmd_iter_info)

    return rmd_iter_info
end

function test_lasso_gd(x★::Vector{Float64}, A::Matrix{Float64}, n::Int64, r::Float64; max_iters::Union{Nothing,Int64}=nothing)
    A_norm = 2 # this holds w.h.p
    prob = LassoProblem(A, A * x★, r,  norm(x★,1), 1 + r * A_norm, n, 1 + log(n))

    gd_iter_info = Iterate_info(:LASSO, x★, 1000, max_iters)
    polyak_subgradient(prob, zeros(n); iter_info = gd_iter_info, max_iters=max_iters)

    return gd_iter_info
end

sizes = [4,5,6]

for size in sizes
    io = open("./results/experiment_2_1e" * string(size) * "_log.txt", "w")
    logger = TeeLogger(ConsoleLogger(io), ConsoleLogger());
    global_logger(logger)

    @info ("LASSO 1e" * string(size) * " test")

    n = Int(10^size)
    k = 5
    r = 3 * sqrt(k)
    K = 25
    @info "n: $n, k: $k, r:$r, K:$K"

    @info "Generating data"
    ratios = [1,2,3,4]
    ms = Int.(ceil.([ratio * (2 * k * log(n / k) + 1.25 * k + 1) for ratio in ratios]))

    x★ = zeros(n)
    x★[1:k] = randn(k)
    x★ ./= norm(x★,1) # not necessary but makes comparison easier
    shuffle(x★)

    A = randn(ms[end],n)

    # warmup
    @info "Warming up"
    test_lasso_rmd(x★, A ./ sqrt(ms[end]), n, r; K = 5, max_iters=10)
    test_lasso_gd(x★, A ./ sqrt(ms[end]), n, r; max_iters=10)

    # real tests
    rmd_iter_infos = Union{Nothing,Iterate_info{:LASSO}}[nothing for _ in 1:length(ms)]
    gd_iter_infos = Union{Nothing,Iterate_info{:LASSO}}[nothing for _ in 1:length(ms)]

    @info "Real tests"
    @info "  RMD, m=$(ms[end])"
    rmd_iter_info = test_lasso_rmd(x★, A ./ sqrt(ms[end]), n, r; K = K)
    rmd_iter_infos[end] = rmd_iter_info
    max_iters = Int(round(1.5 * rmd_iter_info.iter_num[1]))

    for i=1:(length(ms) - 1)
        @info "  RMD, m=$(ms[i])"
        rmd_iter_infos[i] = test_lasso_rmd(x★, A[1:ms[i],:] ./ sqrt(ms[i]), n, r; K = K, max_iters=max_iters)
    end

    for i=1:(length(ms))
        @info "  GD, m=$(ms[i])"
        gd_iter_infos[i] = test_lasso_gd(x★, A[1:ms[i],:] ./ sqrt(ms[i]), n, r; max_iters=max_iters)
    end

    # saving
    @info "Saving data"
    jldsave("./results/experiment_2_1e" * string(size) * ".jld2";
        n=n,
        k=k,
        r=r,
        ms=ms,
        rmd_iter_infos=rmd_iter_infos,
        gd_iter_infos=gd_iter_infos)
end

#===================================================
Plotting
===================================================#

function thin(a,n)
    step = Int(ceil(length(a) / n))
    return a[1:step:end]
end

function plot_experiment(filename, m_labels, out, size)
    results = load(filename) 
    rmd_iter_infos = results["rmd_iter_infos"]
    gd_iter_infos = results["gd_iter_infos"]
    
    Plots.plot()
    colors = palette(:default)[1:4]'
    for i=1:4
        Plots.plot!(
            thin(rmd_iter_infos[i].iter_nums ./ 1000,20), 
            thin(rmd_iter_infos[i].dists,20),
            lw=1.0, linecolor=colors[i],
            label=latexstring("Polyak-RMD ", m_labels[i]))
    end
    for i=1:4
        Plots.plot!(
            thin(gd_iter_infos[i].iter_nums ./ 1000, 20), 
            thin(gd_iter_infos[i].dists, 20), 
            lw=1.0, linestyle=:dash, linecolor=colors[i],
            label=latexstring("Polyak-GD ", m_labels[i]))
    end

    xlabel!("Iteration number (k)")
    yaxis!(:log10)
    Plots.plot!(xtickfontsize=10, xguidefontsize=10, ytickfontsize=10, yguidefontsize=10,leg=Symbol(:outer,:top), legend_columns=4, size=size,ylim=[10^-4, 10])
    Plots.savefig(out)
end

for size in sizes
    plot_experiment("./results/experiment_2_1e" * string(size) * ".jld2", [L"T", L"2T", L"3T", L"4T"], "./results/experiment_2_1e" * string(size) * ".pdf", (500,400))
end

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
Testing
===================================================#

function test_lasso_rmd(x★::Vector{Float64}, A::Matrix{Float64}, n::Int64, r::Float64; K::Int64 = 20, max_iters::Union{Nothing,Int64}=100000)
    A_norm = 2 # this holds w.h.p
    prob = LassoProblem(A, A * x★, r,  norm(x★,1), 1 + r * A_norm, n, 1 + log(n))

    rmd_iter_info = Iterate_info(:LASSO, x★, 1000, max_iters)
    polyak_rmd(prob, zeros(n), K; iter_info = rmd_iter_info)

    return rmd_iter_info
end

function test_lasso_gd(x★::Vector{Float64}, A::Matrix{Float64}, n::Int64, r::Float64; max_iters::Union{Nothing,Int64}=100000)
    A_norm = 2 # this holds w.h.p
    prob = LassoProblem(A, A * x★, r,  norm(x★,1), 1 + r * A_norm, n, 1 + log(n))

    gd_iter_info = Iterate_info(:LASSO, x★, 1000, max_iters)
    polyak_subgradient(prob, zeros(n); iter_info = gd_iter_info, max_iters=max_iters, ϵ = 1e-10)

    return gd_iter_info
end

function run_tests(sizes, k, r, ratio, results_log, out)

    rmd_iter_infos = Union{Nothing,Iterate_info{:LASSO}}[nothing for _ in 1:length(sizes)]
    gd_iter_infos = Union{Nothing,Iterate_info{:LASSO}}[nothing for _ in 1:length(sizes)]

    io = open(results_log, "w")
    logger = TeeLogger(ConsoleLogger(io), ConsoleLogger());
    global_logger(logger)

    for (idx,size) in enumerate(sizes)
        @info ("LASSO 1e" * string(size) * " test")

        n = Int(10^size)
        K = 25
        @info "n: $n, k: $k, r:$r, K:$K"

        @info "Generating data"
        m = Int(ceil(ratio * (2 * k * log(n / k) + 1.25 * k + 1)))

        x★ = zeros(n)
        x★[1:k] = randn(k)
        x★ ./= norm(x★,1) # not necessary but makes comparison easier
        shuffle(x★)

        A = randn(m,n)

        # warmup
        @info "Warming up"
        test_lasso_rmd(x★, A ./ sqrt(m), n, r; K = 5, max_iters=10)
        test_lasso_gd(x★, A ./ sqrt(m), n, r; max_iters=10)

        @info "Real tests"
        @info "  RMD"
        rmd_iter_infos[idx] = test_lasso_rmd(x★, A ./ sqrt(m), n, r; K = K)

        @info "  GD"
        gd_iter_infos[idx] = test_lasso_gd(x★, A ./ sqrt(m), n, r)
    end

    # saving
    @info "Saving data"
    jldsave(out;
        ns = [Int(10^size) for size in sizes],
        k=k,
        r=r,
        rmd_iter_infos=rmd_iter_infos,
        gd_iter_infos=gd_iter_infos)
end

#===================================================
Plotting
===================================================#

function plot_experiment(filename, out_rmd, out_gd; size=(400,300), exp_spacing=1.1)
    results = load(filename)
    rmd_iter_infos = results["rmd_iter_infos"]
    gd_iter_infos = results["gd_iter_infos"]
    ns = results["ns"]
    num_ns = length(ns)
    
    idxs = unique(Int.(round.([exp_spacing^k for k = 1:Int(round(log2(10^5)/log2(exp_spacing)))])))

    for (out, iter_infos) in [(out_rmd, rmd_iter_infos), (out_gd, gd_iter_infos)]
        Plots.plot()
        colors = palette(:default)[1:num_ns]'
        for (i, n) in enumerate(ns)
            iters = iter_infos[i].iter_nums
            dists = iter_infos[i].dists

            iters = iters[[i for i in idxs if i <= length(iters)]]
            dists = dists[[i for i in idxs if i <= length(dists)]]

            Plots.plot!(
                iters, 
                dists,
                lw=1.0, linecolor=colors[i],
                label=latexstring("n = ", n))
        end

        xlabel!("Iteration number")
        ylabel!("Distance")

        xaxis!(:log10)
        yaxis!(:log10)

        Plots.plot!(xtickfontsize=10, xguidefontsize=10, ytickfontsize=10, yguidefontsize=10,leg=Symbol(:outer,:top), legend_columns=num_ns, size=size,ylim=[10^-5, 10], xlim=[1,10^5])
        Plots.savefig(out)
    end
end

#===================================================
Script
===================================================#

# Test parameters
sizes = [3,4,5,6]
k = 5
r = 3 * sqrt(k)
ratio = 2

# Save locations
results_log = "./results/experiment_log.txt"
results_data = "./results/experiment.jld2"
rmd_plot = "./results/rmd_plot.pdf"
gd_plot = "./results/polyak_gd_plot.pdf"

#   
run_tests(sizes, k, r, ratio, results_log, results_data)
plot_experiment(results_data, rmd_plot, gd_plot)
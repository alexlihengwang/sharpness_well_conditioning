using LinearAlgebra
using Arpack
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

function test_lasso(x★::Vector{Float64}, A::Matrix{Float64}, n::Int64, r::Float64; K::Int64 = 20, max_iters::Union{Nothing,Int64}=nothing)
    A_norm = 2 # this holds w.h.p
    prob = LassoProblem(A, A * x★, r,  norm(x★,1), 1 + r * A_norm, n, 1 + log(n))

    iter_info = Iterate_info(:LASSO, x★, 1000, max_iters)
    polyak_rmd(prob, zeros(n), K; iter_info = iter_info)

    return iter_info
end

io = open("./results/experiment_1_lasso_log.txt", "w")
logger = TeeLogger(ConsoleLogger(io), ConsoleLogger());
global_logger(logger)

@info "LASSO test"

n = 10000
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
test_lasso(x★, A ./ sqrt(ms[end]), n, r; K = 5, max_iters=10)

# real tests
iter_infos = Union{Nothing,Iterate_info{:LASSO}}[nothing for _ in 1:length(ms)]

@info "Real tests"
@info "  m=$(ms[end])"
iter_info = test_lasso(x★, A ./ sqrt(ms[end]), n, r; K = K)
iter_infos[end] = iter_info
max_iters = Int(round(1.5 * iter_info.iter_num[1]))

for i=1:(length(ms) - 1)
    @info "  m=$(ms[i])"
    iter_infos[i] = test_lasso(x★, A[1:ms[i],:] ./ sqrt(ms[i]), n, r; K = K, max_iters=max_iters)
end

# saving
@info "Saving data"
jldsave("./results/experiment_1_lasso.jld2";
    n=n,
    k=k,
    ms=ms,
    iter_infos=iter_infos)

#===================================================
Matrix sensing problems
===================================================#

function test_matrix_sensing(X★::Matrix{Float64}, As::Matrix{Float64}, n₁::Int64, n₂::Int64, r::Float64; K::Int64 = 20, max_iters::Union{Nothing,Int64}=nothing)
    n = minimum([n₁,n₂])

    b = As * vec(X★)
    _, X★Σ, _ = svd(X★)

    𝒜_norm = 2 # this holds w.h.p.

    prob = MatrixSensingProblem(As, b, r, sum(X★Σ), 1 + r * 𝒜_norm, n, 1 + log(n))
    iter_info = Iterate_info(:MATSENS, X★, 100, max_iters)
    polyak_rmd(prob, zeros(n₁,n₂), K; iter_info = iter_info)
    return iter_info
end

io = open("./results/experiment_1_matsens_log.txt", "w")
logger = TeeLogger(ConsoleLogger(io), ConsoleLogger());
global_logger(logger)

@info "MATSENS test"

n₁ = 100
n₂ = 100
n = minimum([n₁,n₂])
k = 5
r = 3 * sqrt(k)
K = 25
@info "n₁×n₂: $n₁×$n₂, k: $k, r:$r, K:$K"

@info "Generating data"
ratios = [1,2,3,4]
ms = Int.(ceil.([ratio * (3 * k * (n₁+ n₂ - k) + 1) for ratio in ratios]))

X★ = randn(n₁,n₂)
X★U, X★Σ, X★V = svds(X★, nsv = k)[1];
X★Σ ./= norm(X★Σ,1) # not necessary but makes comparison easier
X★ = X★U * (X★Σ .* X★V')

As = randn(ms[end],n₁*n₂)

# warmup
@info "Warming up"
test_matrix_sensing(X★, As ./ sqrt(ms[end]), n₁, n₂, r; K=5)

# real tests
iter_infos = Union{Nothing,Iterate_info{:MATSENS}}[nothing for _ in 1:length(ms)]

@info "Real tests"
@info "  m=$(ms[end])"
iter_info = test_matrix_sensing(X★, As ./ sqrt(ms[end]), n₁, n₂, r; K=K)
iter_infos[end] = iter_info
max_iters = Int(round(1.5 * iter_info.iter_num[1]))

for i=1:(length(ms) - 1)
    @info "  m=$(ms[i])"
    iter_infos[i] = test_matrix_sensing(X★, As[1:ms[i],:] ./ sqrt(ms[i]), n₁, n₂, r; K=K, max_iters=max_iters)
end

# saving
@info "Saving data"
jldsave("./results/experiment_1_matsens.jld2";
    n₁=n₁,
    n₂=n₂,
    k=k,
    ms=ms,
    iter_infos=iter_infos)


#===================================================
Phase retrieval problems
===================================================#

function test_phase_retrieval(X★::Matrix{Float64}, G::Matrix{Float64}, n::Int64, r₁::Float64, r₂::Float64; K::Int64 = 20, max_iters::Union{Nothing,Int64}=nothing)
    G_norm = 2 # this holds w.h.p.
    b = vec(sum((G * X★) .* G,dims=2))

    prob = PhaseRetrievalProblem(G, b, r₁, r₂, tr(X★), 1 + r₁ + r₂ * G_norm, n, 1 + log(n))
    iter_info = Iterate_info(:PHRET, X★, 100, max_iters)
    polyak_rmd(prob, zeros(n,n), K; iter_info = iter_info)
    return iter_info
end

io = open("./results/experiment_1_phret_log.txt", "w")
logger = TeeLogger(ConsoleLogger(io), ConsoleLogger());
global_logger(logger)

@info "PHRET test"

n = 100
r₁ = 2.0
r₂ = 3.0
K = 25
@info "n: $n, r₁: $r₁, r₂:$r₂, K:$K"

@info "Generating data"
ratios = [4,8,16,32,64]
ms = Int.(ceil.([ratio * 2 * n for ratio in ratios]))

x★ = randn(n)
normalize!(x★) # not necessary but makes comparison easier
X★ = x★ * x★'
G = randn(ms[end], n)

# warmup
@info "Warming up"
test_phase_retrieval(X★, G ./ sqrt(ms[end]), n, r₁, r₂; K = 5)

iter_infos = Union{Nothing,Iterate_info{:PHRET}}[nothing for _ in 1:length(ms)]

@info "Real tests"
@info "  m=$(ms[end])"
iter_info = test_phase_retrieval(X★, G ./ sqrt(ms[end]), n, r₁, r₂; K = K)
iter_infos[end] = iter_info
max_iters = Int(round(1.5 * iter_info.iter_num[1]))

for i=1:(length(ms) - 1)
    @info "  m=$(ms[i])"
    iter_infos[i] = test_phase_retrieval(X★, G[1:ms[i],:] ./ sqrt(ms[i]), n, r₁, r₂; K = K, max_iters=max_iters)
end

# saving
@info "Saving data"
jldsave("./results/experiment_1_phret.jld2";
    n=n,
    ms=ms,
    iter_infos=iter_infos)


#===================================================
Plotting
===================================================#

function thin(a,n)
    step = Int(ceil(length(a) / n))
    return a[1:step:end]
end

function plot_experiment(filename, m_labels, out, size)
    results = load(filename) 
    iter_infos = results["iter_infos"]
    
    Plots.plot()
    for i=1:4
        Plots.plot!(
            thin(iter_infos[i].iter_nums ./ 1000,20), 
            thin(iter_infos[i].dists, 20),
            lw=1.0, label=m_labels[i])
    end
    
    xlabel!("Iteration number (k)")
    yaxis!(:log10)
    Plots.plot!(xtickfontsize=10, xguidefontsize=10, ytickfontsize=10, yguidefontsize=10,leg=Symbol(:outer,:top), legend_column=-1, size=size)
    Plots.savefig(out)
end

plot_experiment("./results/experiment_1_lasso.jld2", [L"T", L"2T", L"3T", L"4T"], "./results/experiment_1_lasso.pdf", (500,400))
plot_experiment("./results/experiment_1_matsens.jld2", [L"T", L"2T", L"3T", L"4T"], "./results/experiment_1_matsens.pdf", (500,400))
plot_experiment("./results/experiment_1_phret.jld2", [L"4T", L"8T", L"16T", L"32T"], "./results/experiment_1_phret.pdf", (500,400))

using LinearAlgebra

#===================================================
Mirror descent and Polyak-RMD
===================================================#

# Problem input/output structs
abstract type MDProblem end
function f_oracle(prob::MDProblem, x̄::Array{Float64})
    @warn "f_oracle undefined"
end
function mirror(prob::MDProblem, θ::Array{Float64})
    # must implement θ ↦ sign(θ) ∘ | θ |^(q-1) / ‖ θ ‖_q^(q-2)
    @warn "f_oracle undefined"
end

struct Iterate_info{problem_symbol}
    x★::Array{Float64} # need optimal solution
    iter_num::Vector{Int64} # what iteration number currently on
    iter_freq::Int64 # how often to log solution
    time_0::Float64 # time when object created
    times::Vector{Float64}
    dists::Vector{Float64}
    iter_nums::Vector{Int64}
    max_total_iters::Union{Int64,Nothing}
    ts_per_round::Vector{Int64}
end

function Iterate_info(problem_symbol::Symbol, x★::Array{Float64}, freq::Int64,max_total_iters::Union{Int64,Nothing})
    Iterate_info{problem_symbol}(x★, Int64[0], freq, time(), Float64[], Float64[], Int64[], max_total_iters, Int64[])
end

# MD and RMD alorithms
# In this implementation, we assume that f has been shifted so that min_x f(x) = 0

function mirror_descent(prob::MDProblem, x̄::Array{Float64}; maxiter::Int64=100000, iter_info::Union{Iterate_info,Nothing}=nothing, α::Float64=exp(1/2))
    ϵ₀, _ = f_oracle(prob, x̄)
    η = ϵ₀ / (exp(2) * prob.L^2 * log(prob.n))

    val = nothing
    x = copy(x̄)
    θ = zeros(size(x̄))

    t = 1
    while true
        val, g = f_oracle(prob, x)

        if !isnothing(iter_info)
            push_iterate!(iter_info, x) || break
        end

        if α * val <= ϵ₀
            !isnothing(iter_info) && push!(iter_info.ts_per_round, t)
            return val, x, true
        end

        θ .-= η .* g
        x = x̄ .+ mirror(prob, θ)

        t += 1
    end
    !isnothing(iter_info) && push!(iter_info.ts_per_round, t)
    return val, x, false
end

function polyak_rmd(prob::MDProblem, x̄::Array{Float64}, K::Int64; iter_info::Union{Iterate_info,Nothing}=nothing, α::Float64=exp(1/2))
    # Implements Polyak-RMD (Algorithm 3)
    !isnothing(iter_info) && push_iterate!(iter_info, x̄)
    for k=1:K
        val, x̄, flag = mirror_descent(prob, x̄; iter_info=iter_info, α=α)
        flag || break
    end

    return x̄
end

# utility functions

function sign_power(x::Float64, p::Float64)
    return sign(x) * abs(x)^p
end

# polyak step sharp function

function polyak_subgradient(prob::MDProblem, x̄::Array{Float64}; max_iters::Int64=10000, λ=1.0, iter_info::Union{Iterate_info,Nothing}=nothing, ϵ = 1e-8)
    # Implements subgradient descent with Polyak step-size
    
    x = copy(x̄)
    !isnothing(iter_info) && push_iterate!(iter_info, x)

    for t = 1:max_iters
        val, g = f_oracle(prob, x)
        x .-= (λ * val / (norm(g)^2)) .* g

        if !isnothing(iter_info)
            push_iterate!(iter_info, x) || break
        end
        
        if val < ϵ
            break
        end
    end

    return x
end

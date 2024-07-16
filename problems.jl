using LinearAlgebra

#===================================================
LASSO problems
===================================================#

struct LassoProblem <: MDProblem
    A::Matrix{Float64}
    b::Vector{Float64}
    r::Float64
    opt_val::Float64
    L::Float64
    n::Int64
    q::Float64
end

function f_oracle(prob::LassoProblem, x::Vector{Float64})
    # first order oracle for:
    # |x|₁ + r |Ax - b|₂ - |x^*|₁

    diff = prob.A * x .- prob.b
    val = norm(x, 1) + prob.r * norm(diff) - prob.opt_val

    grad = sign.(x)
    if norm(diff) > eps(Float64)
        normalize!(diff)
        grad .+= prob.r .* (prob.A' * diff)
    end

    return val, grad
end

function mirror(prob::LassoProblem, θ::Vector{Float64})
    q = prob.q
    if norm(θ, q) < eps(Float64)
        return zeros(size(θ))
    else
        return sign_power.(θ, q - 1) ./ (norm(θ, q)^(q-2))
    end
end

function push_iterate!(iter_info::Iterate_info{:LASSO}, x::Vector{Float64})
    push!(iter_info.times, time() - iter_info.time_0)
    push!(iter_info.dists, norm(iter_info.x★ .- x,1))
    push!(iter_info.iter_nums, iter_info.iter_num[1])
    if (iter_info.iter_num[1] % iter_info.iter_freq) == 0
        @info "    iterate number: $(iter_info.iter_num[1])"
        @info "    distance: $(iter_info.dists[end])"
    end
    iter_info.iter_num[1] += 1

    if iter_info.dists[end] < 10^-5
        return false
    end

    if isnothing(iter_info.max_total_iters)
        return true
    else
        return iter_info.max_total_iters > iter_info.iter_num[1]
    end
end
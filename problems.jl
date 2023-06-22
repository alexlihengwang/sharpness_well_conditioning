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

    if isnothing(iter_info.max_total_iters)
        return true
    else
        return iter_info.max_total_iters > iter_info.iter_num[1]
    end
end

#===================================================
Matrix sensing problems
===================================================#

struct MatrixSensingProblem <: MDProblem
    As::Matrix{Float64}
    b::Vector{Float64}
    r::Float64
    opt_val::Float64
    L::Float64
    n::Int64
    q::Float64
end

function f_oracle(prob::MatrixSensingProblem, X::Matrix{Float64})
    # first order oracle for:
    # |X|₁ + r |𝒜(X) - b|₂ - |X^*|₁

    XU, XΣ, XV = svd(X)

    diff = prob.As * vec(X)
    diff .-= prob.b
    norm_diff = norm(diff)

    val = sum(XΣ) + prob.r * norm_diff - prob.opt_val

    grad = XU * XV'
    if norm_diff > eps(Float64)
        grad .+= prob.r .* reshape(prob.As' * (diff ./ norm_diff), size(X))
    end

    return val, grad
end

function mirror(prob::MatrixSensingProblem, θ::Matrix{Float64})
    q = prob.q

    θU, θΣ, θV = svd(θ)

    if norm(θΣ, q) < eps(Float64)
        return zeros(size(θ))
    else
        return θU * ((sign_power.(θΣ, q - 1) ./ (norm(θΣ, q)^(q-2))) .* θV')
    end
end

function push_iterate!(iter_info::Iterate_info{:MATSENS}, X::Matrix{Float64})
    if (iter_info.iter_num[1] % iter_info.iter_freq) == 0
        diff = X .- iter_info.x★
        _, diffΣ, _ = svd(diff)
        push!(iter_info.times, time() - iter_info.time_0)
        push!(iter_info.dists, norm(diffΣ,1))
        push!(iter_info.iter_nums, iter_info.iter_num[1])
        @info "    iterate number: $(iter_info.iter_num[1])"
        @info "    distance: $(iter_info.dists[end])"
    end
    iter_info.iter_num[1] += 1

    if isnothing(iter_info.max_total_iters)
        return true
    else
        return iter_info.max_total_iters > iter_info.iter_num[1]
    end
end

#===================================================
Phase retrieval problems
===================================================#

# definitions

function sym!(X::Matrix{Float64})
    X .+= X'
    X ./= 2
end

struct PhaseRetrievalProblem <: MDProblem
    G::Matrix{Float64} # m×n matrix of sensing vectors
    b::Vector{Float64}
    r₁::Float64
    r₂::Float64
    opt_val::Float64
    L::Float64
    n::Int64
    q::Float64
end

function f_oracle(prob::PhaseRetrievalProblem, X::Matrix{Float64})
    # first order oracle for:
    # tr(X) + r₁ tr(X₋) + r₂ ‖ diag(GXG') - b ‖₁ - tr(X★)
    vals, vecs = eigen(Symmetric(X))

    diff = vec(sum((prob.G * X) .* prob.G,dims=2))
    diff .-= prob.b

    val = tr(X) - prob.r₁ * sum(vals .* (vals .< 0)) + prob.r₂ * norm(diff,1) - prob.opt_val

    grad = Matrix(1.0I, size(X)...)
    grad .-= prob.r₁ .* (vecs * ((vals .< 0) .* vecs'))
    grad .+= prob.r₂ .* (prob.G' * (sign.(diff) .* prob.G))

    return val, grad
end

function mirror(prob::PhaseRetrievalProblem, θ::Matrix{Float64})
    sym!(θ)
    vals, vecs = eigen(Symmetric(θ))
    X = vecs * ((sign_power.(vals, prob.q - 1) ./ (norm(vals, prob.q)^(prob.q - 2))) .* vecs')
    sym!(X)

    return X
end

function push_iterate!(iter_info::Iterate_info{:PHRET}, X::Matrix{Float64})
    if (iter_info.iter_num[1] % iter_info.iter_freq) == 0
        diff = X .- iter_info.x★
        vals = eigvals(Symmetric(diff))
        push!(iter_info.times, time() - iter_info.time_0)
        push!(iter_info.dists, norm(vals,1))
        push!(iter_info.iter_nums, iter_info.iter_num[1])
        @info "    iterate number: $(iter_info.iter_num[1])"
        @info "    distance: $(iter_info.dists[end])"
    end
    iter_info.iter_num[1] += 1

    if isnothing(iter_info.max_total_iters)
        return true
    else
        return iter_info.max_total_iters > iter_info.iter_num[1]
    end
end

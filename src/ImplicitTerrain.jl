module ImplicitTerrain

using ADTypes, Lux, Random, Optimisers, StyledStrings, Rasters, DataFrames, Images, ProgressMeter
using ..GeoSurrogates: is_normalized, normalize

import StatsAPI: predict, fit!
import DimensionalData as DD

# Initialization weights for first layer
init_weight_1(rng, out, in) = (2rand(rng, Float32, out, in) .- 1) ./ in

max_init_weight(in) = sqrt(6f0 ./ in)

init_weight_h(rng, out, in) = (max_init_weight(in) * rand(rng, Float32, out, in) .- 1) ./ in

init_bias(rng, out) = zeros(Float32, out)

# SIREN activation functor (avoids closure capture overhead)
struct SIRENActivation{T} <: Function
    ω::T
end
(s::SIRENActivation)(x) = sin(s.ω * x)

# NNlib compatibility for fast activation
import NNlib: fast_act
fast_act(s::SIRENActivation, ::AbstractArray) = s

#-----------------------------------------------------------------------------# MLP
mutable struct MLP{T, P, S, TS}
    chain::T
    parameters::P
    states::S
	train_state::TS

    function MLP(;
                in = 2,
                hidden = 256,
                out = 1,
                n_hidden = 3,
                ω0 = 30f0,
                ωh = 1f0,
                rng = Random.MersenneTwister(42),
                alg = Adam(0.0001f0),
                init_weight_1 = init_weight_1,
                init_weight_h = init_weight_h,
                init_bias = init_bias
			)
	    layers = [
	        Dense(in, hidden, SIRENActivation(ω0); init_weight = init_weight_1, init_bias),
	        [Dense(hidden, hidden, sin; init_weight = init_weight_h, init_bias) for _ in 2:n_hidden]...,
	        Dense(hidden, out, sin; init_weight=init_weight_h, init_bias)
	    ]
        model = Chain(layers...)
        ps, st = Lux.setup(rng, model)
		ts = Lux.Training.TrainState(model, ps, st, alg)
        new{typeof(model), typeof(ps), typeof(st), typeof(ts)}(model, ps, st, ts)
    end
end

Base.show(io::IO, M::MIME"text/plain", o::MLP) = print(io, "ImplicitTerrain.MLP")

predict(o::MLP, x::AbstractMatrix) = Lux.apply(o.chain, x, o.train_state.parameters, o.train_state.states)[1]

predict(o::MLP, coords::Tuple) = predict(o, hcat(collect(Float32, coords)))[1]

function predict(o::MLP, r::Raster)
    data = features(r)
    ẑ = predict(o, data)
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

function features(r::Raster)
    # Extract coordinates directly from raster dimensions (avoid DataFrame conversion)
    xs = Float32.(DD.dims(r, X).val)
    ys = Float32.(DD.dims(r, Y).val)
    nx, ny = length(xs), length(ys)
    # Create coordinate grid: each column is (x, y) for one pixel
    coords = Matrix{Float32}(undef, 2, nx * ny)
    idx = 1
    for j in 1:ny, i in 1:nx
        coords[1, idx] = xs[i]
        coords[2, idx] = ys[j]
        idx += 1
    end
    return coords
end

#-----------------------------------------------------------------------------# fit!
function fit!(o::MLP, x::AbstractMatrix, y::AbstractVector; steps = 1, batchsize = nothing)
    ts = o.train_state
    loss_fn = Lux.MSELoss()
    backend = AutoZygote()
    n = size(x, 2)
    bs = isnothing(batchsize) ? n : min(batchsize, n)

    if bs >= n
        # Full-batch training
        y_row = y'
        for _ in 1:steps
            _, _, _, ts = Lux.Training.single_train_step!(backend, loss_fn, (x, y_row), ts)
        end
    else
        # Mini-batch training
        for _ in 1:steps
            idxs = rand(1:n, bs)
            x_batch = @view x[:, idxs]
            y_batch = y[idxs]'
            _, _, _, ts = Lux.Training.single_train_step!(backend, loss_fn, (x_batch, y_batch), ts)
        end
    end
    o.train_state = ts
    return o
end

function fit!(o::MLP, r::Raster; steps = 1, batchsize = nothing)
    is_normalized(r) || error("Raster must be normalized (see GeoSurrogates.normalize) before fitting.")
    x = features(r)
    y = r.data isa AbstractArray{Float32} ? vec(r.data) : Float32.(vec(r.data))
    return fit!(o, x, y; steps, batchsize)
end


#-----------------------------------------------------------------------------# Model
struct Model{S, G}
    surface::S
    geometry::G
end
Model() = Model(MLP(), MLP())

Base.show(io::IO, M::MIME"text/plain", o::Model) = print(io, "ImplicitTerrain.Model")

function predict(o::Model, x)
    ẑs = predict(o.surface, x)
    ẑg = predict(o.geometry, x)
    return ẑs + ẑg
end

#-----------------------------------------------------------------------------# fit!
# Here steps == steps per pyramid level
function fit!(o::Model, r::Raster; steps = 1000, batchsize = nothing)
    pyr = _gaussian_pyramid(r)
    # Fit surface model progressively on pyramid levels (coarse to fine)
    for (i, level) in enumerate(reverse(@view(pyr[2:end])))
        @info "Fitting surface model on pyramid level $i/$(length(pyr) - 1) with size $(size(level))"
        fit!(o.surface, level; steps, batchsize)
    end
    # Compute residuals: original - surface_prediction
    full_res = first(pyr)
    x = features(full_res)
    surface_pred = predict(o.surface, x)
    y_residuals = Float32.(vec(full_res.data)) .- vec(surface_pred)
    # Fit geometry model on residuals to capture fine details
    @info "Fitting geometry model on residuals with size $(size(full_res))" residual_range=extrema(y_residuals)
    fit!(o.geometry, x, y_residuals; steps, batchsize)
    return o
end

#-----------------------------------------------------------------------------# _gaussian_pyramid
# Avoiding piracy from Images/Rasters
# Assumes input raster is normalized
function _gaussian_pyramid(r::Raster, σ = 4, n = 4, downsample = 2)
    # Preserve dimension ordering (increasing vs decreasing)
    x_order = step(DD.dims(r, X).val.data) > 0 ? (-1f0, 1f0) : (1f0, -1f0)
    y_order = step(DD.dims(r, Y).val.data) > 0 ? (-1f0, 1f0) : (1f0, -1f0)
    dims = map(DD.dims(r)) do d
        if DD.Lookups.iscategorical(d)
            return d
        else
            order = d isa DD.Dimensions.X ? x_order : y_order
            x = range(order[1], order[2], length=length(d))
            return DD.rebuild(d, x)
        end
    end
    rdata = r.data isa AbstractArray{Float32} ? r.data : Float32.(r.data)
    pyramid = [Raster(rdata, dims)]

    for i in 2:n
        prev = pyramid[end]
        filtered = Float32.(imfilter(prev, Kernel.gaussian(σ)))
        resampled = resample(filtered, method = :average, size = size(prev) .÷ downsample)
        # Put back into Float32, preserving dimension order:
        x = range(x_order[1], x_order[2], length=size(resampled, 1))
        y = range(y_order[1], y_order[2], length=size(resampled, 2))
        dims = (X(x), Y(y))
        next = Raster(Float32.(resampled.data), dims)
        push!(pyramid, next)
    end
    return pyramid
end


# #-----------------------------------------------------------------------------# ImplicitTerrainTrainer
# @kwdef struct ImplicitTerrainTrainer
#     raster::Raster
#     pyramid::Vector{Raster} = _gaussian_pyramid(normalize(raster))
#     alg = Adam(0.001f0)
#     loss = MSELoss()
#     chain_s::ImplicitTerrainChain = ImplicitTerrainChain()
#     train_state_s = Lux.Training.TrainState(chain_s.chain, chain_s.parameters, chain_s.states, alg)
#     chain_g::ImplicitTerrainChain = ImplicitTerrainChain()
#     train_state_g = Lux.Training.TrainState(chain_g.chain, chain_g.parameters, chain_g.states, alg)
#     backend::AbstractADType = AutoZygote()
# end
# Base.show(io::IO, M::MIME"text/plain", o::ImplicitTerrainTrainer) =
#     print(io, styled"{bright_cyan:ImplicitTerrainTrainer} with raster of size $(size(o.raster))")

# function train_surface_model!(o::ImplicitTerrainTrainer; steps = 3000)
#     steps_per_level = steps ÷ (length(o.pyramid) - 1)
#     (; alg, loss, backend) = o
#     R = o.pyramid[1]
#     x, y = xyfeatures(R)
#     X = Float32.(vcat(x', y'))
#     z = Float32.(vec(R.data)')

#     (; chain, parameters, states) = o.chain_s

#     train_state_s = o.train_state_s

#     for level in reverse(@view(o.pyramid[2:end]))
#         @info "Training on pyramid level with size $(size(level))"
#         for i in 1:steps_per_level
#             Lux.Training.single_train_step!(backend, loss, (X, z), train_state_s)
#         end
#     end

#     o
# end

# function predict_surface(o::ImplicitTerrainTrainer, x::AbstractVector, y::AbstractVector)
#     (; chain, parameters, states) = o.chain_s
#     x_norm = Float32.(x)
#     y_norm = Float32.(y)
#     XY = Float32.(vcat(x_norm', y_norm'))
#     z_pred, _ = chain.chain(XY, parameters, states)
#     return vec(z_pred)
# end

# # function train_geometry_model!(o::ImplicitTerrainTrainer; steps = 3000)
# #     residuals = o.raster .- o.chain_s
# # end


# # normalize x, y, and z to [-1, 1]
# # Both Models: 3000 steps via Adam optimizer with learning rate 1e-4
# # Gaussian pyramid: 4 levels, σ = 4.0

# # Input: 1000 x 1000 raster
# # Φs: Fit progressively on top 3 layers of 4-layer gaussian pyramid.  1000 steps per layer (?) to get 3000.
# # Φg: Fit on residuals of (original_image_pixel - Φs(predicted_pixel))


# #-----------------------------------------------------------------------------# GaussianPyramid





# #-----------------------------------------------------------------------------# ImplicitTerrain
# #=
# ImplicitTerrain: A Continuous Surface Model for Terrain Data Analysis

# Implementation based on: https://arxiv.org/html/2406.00227

# Architecture:
# - Two cascaded MLPs (Surface Model and Geometry Model)
# - 3 hidden layers each with 256 units
# - SIREN activation functions (sinusoidal)
# - Input: 2D coordinates (x, y) normalized to [-1, 1]
# - Output: Height value z (scalar)
# =#

# @kwdef struct ImplicitTerrain
#     x_extrema::Tuple{Float32, Float32}
#     y_extrema::Tuple{Float32, Float32}
#     crs::Any
#     rng::AbstractRNG = MersenneTwister(42)
#     surface_model::Lux.Chain = mlp_chain()
#     surface_params::NamedTuple
#     surface_state::NamedTuple
#     geometry_model::Lux.Chain = mlp_chain()
#     geometry_params::NamedTuple
#     geometry_state::NamedTuple
# end


end  # module ImplicitTerrain

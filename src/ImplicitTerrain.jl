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

#-----------------------------------------------------------------------------# MLP
struct MLP{T, P, S, TS}
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
	        Dense(in, hidden, x -> sin(ω0 * x); init_weight = init_weight_1, init_bias),
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

predict(o::MLP, x::AbstractMatrix) = Lux.apply(o.chain, x, o.parameters, o.states)[1]

predict(o::MLP, coords::Tuple) = predict(o, hcat(collect(Float32, coords)))[1]

function predict(o::MLP, r::Raster)
    data = features(r)
    ẑ = predict(o, data)
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

function features(r::Raster)
    df = DataFrame(r)
    x = normalize(df.X)
    y = normalize(df.Y)
    return Float32[x'; y']
end

#-----------------------------------------------------------------------------# fit!
function fit!(o::MLP, x::AbstractMatrix, y::AbstractVector; steps = 1)
    @showprogress for _ in 1:steps
        Lux.Training.single_train_step!(AutoZygote(), Lux.MSELoss(), (x, y'), o.train_state)
    end
    return o
end

function fit!(o::MLP, r::Raster; steps = 1)
    is_normalized(r) || error("Raster must be normalized (see GeoSurrogates.normalize) before fitting.")
    x = features(r)
    y = Float32.(vec(r.data))
    return fit!(o, x, y; steps=steps)
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
function fit!(o::Model, r::Raster; steps = 1000)
    pyr = _gaussian_pyramid(r)
    for (i, level) in enumerate(reverse(@view(pyr[2:end])))
        @info "Fitting surface model on pyramid level $i/$(length(pyr) - 1) with size $(size(level))"
        fit!(o.surface, level; steps)
    end
    @info "Fitting geometry model on original raster with size $(size(r))"
    fit!(o.geometry, first(pyr); steps)
    return o
end

#-----------------------------------------------------------------------------# _gaussian_pyramid
# Avoiding piracy from Images/Rasters
# Assumes input raster is normalized
function _gaussian_pyramid(r::Raster, σ = 4, n = 4, downsample = 2)
    rnorm = normalize(r)
    dims = map(DD.dims(r)) do d
        if DD.Lookups.iscategorical(d)
            return d
        else
            x = range(-1f0, 1f0, length=length(d))
            return DD.rebuild(d, x)
        end
    end
    pyramid = [Raster(Float32.(rnorm.data), dims)]

    for i in 2:n
        prev = pyramid[end]
        filtered = Float32.(imfilter(prev, Kernel.gaussian(σ)))
        resampled = resample(filtered, method = :average, size = size(prev) .÷ downsample)
        # Put back into Float32:
        x = range(-1f0, 1f0, length=size(resampled, 1))
        y = range(-1f0, 1f0, length=size(resampled, 2))
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

module ImplicitTerrain

using ADTypes, Lux, Random, Optimisers, StyledStrings, Rasters, DataFrames

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
                alg = Adam(0.001f0),
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
    df = DataFrame(r)
    data = Float32[df.X'; df.Y']
    ẑ = predict(o, data)
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
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



# #-----------------------------------------------------------------------------# ImplicitTerrainChain
# # Dense MLP with specified architecture:
# # Note: SIREN paper is missing details on hyperparameters ω.  Defaults are set according to notes
# #       on the implementation on GitHub.
# function implicit_terrain_chain(;
#         rng = Random.MersenneTwister(42),
#         in = 2,
#         hidden = 256,
#         out = 1,
#         n_hidden = 3,
#         ω0 = Float32(30),  # First layer: sin(ω0 * W * x + b)
#         ωh = Float32(1)
#     )
#     init_weight_1(rng, out, in) = rand(rng, Float32, out, in) .* 2 ./ in .- 1 ./ in
#     init_weight_h(rng, out, in) = rand(rng, Float32, out, in) .* sqrt(6 ./ in) ./ ω0 .- sqrt(6 ./ in) ./ ω0
#     init_bias = (rng, out) -> zeros(Float32, out)
#     layers = [
#         Dense(in, hidden, x -> sin(ω0 * x); init_weight = init_weight_1, init_bias),
#         [Dense(hidden, hidden, sin; init_weight = init_weight_h, init_bias) for _ in 2:n_hidden]...,
#         Dense(hidden, out, sin; init_weight=init_weight_h, init_bias)
#     ]
#     return Chain(layers...)
# end


# struct ImplicitTerrainChain{T, P, S}
#     chain::T
#     parameters::P
#     states::S

#     function ImplicitTerrainChain(; rng = Random.MersenneTwister(42), kw...)
#         model = implicit_terrain_chain(; rng, kw...)
#         ps, st = Lux.setup(rng, model)
#         new{typeof(model), typeof(ps), typeof(st)}(model, ps, st)
#     end
# end

# function Base.show(io::IO, M::MIME"text/plain", o::ImplicitTerrainChain)
#     println(io, styled"{bright_cyan:ImplicitTerrainChain}:")
#     Base.show(io, M, o.chain)
# end

# #-----------------------------------------------------------------------------# gaussian_pyramid
# # Avoiding piracy from Images/Rasters
# # Assumes input raster is normalized
# function _gaussian_pyramid(r::Raster, σ = 4, n = 4, downsample = 2)
#     is_normalized(r) || error("Input raster must be normalized via GeoSurrogates.normalize")
#     pyramid = [r]
#     for i in 2:n
#         prev = pyramid[end]
#         filtered = Float32.(imfilter(prev, Kernel.gaussian(σ)))
#         resampled = resample(filtered, method = :average, size = size(prev) .÷ downsample)
#         # Put back into Float32:
#         x = range(-1f0, 1f0, length=size(resampled, 1))
#         y = range(-1f0, 1f0, length=size(resampled, 2))
#         dims = (X(x), Y(y))
#         next = Raster(Float32.(resampled.data), dims)
#         push!(pyramid, next)
#     end
#     return pyramid
# end


# #-----------------------------------------------------------------------------# normalize
# # Normalize raster values to [-1, 1]
# function normalize(r::Raster{T, 2}) where {T}
#     x = range(-1f0, 1f0, length=size(r, 1))
#     y = range(-1f0, 1f0, length=size(r, 2))
#     a, b = extrema(r)
#     z = Float32.(normalize.(r.data, a, b))
#     dims = (X(x), Y(y))
#     return Raster(z, dims)
# end

# function is_normalized(r::Raster)
#     a, b = extrema(r)
#     xa, xb = extrema(r.dims[1])
#     ya, yb = extrema(r.dims[2])
#     a >= -1f0 && b <= 1f0 && xa >= -1f0 && xb <= 1f0 && ya >= -1f0 && yb <= 1f0
# end



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

# normalize(x, min, max) = 2one(x) * (x .- min) ./ (max - min) .- one(x)
# normalize(x, y, o::ImplicitTerrain) = normalize(x, o.extent.X[1], o.extent.X[2]), normalize(y, o.extent.Y[1], o.extent.Y[2])

# function Lux.initialparameters(rng::AbstractRNG, layer::SIRENLayer)
#     # SIREN initialization scheme
#     if layer.is_first
#         # First layer: uniform(-1/n, 1/n)
#         bound = 1.0f0 / layer.in_dims
#     else
#         # Other layers: uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
#         bound = sqrt(6.0f0 / layer.in_dims) / layer.omega_0
#     end

#     W = rand(rng, Float32, layer.out_dims, layer.in_dims) .* 2f0 .* bound .- bound
#     b = rand(rng, Float32, layer.out_dims) .* 2f0 .* bound .- bound

#     return (weight=W, bias=b)
# end

# Lux.initialstates(::AbstractRNG, ::SIRENLayer) = NamedTuple()

# function (layer::SIRENLayer)(x, ps, st)
#     y = ps.weight * x .+ ps.bias
#     return sin.(layer.omega_0 .* y), st
# end

# Lux.parameterlength(layer::SIRENLayer) = layer.in_dims * layer.out_dims + layer.out_dims
# Lux.statelength(::SIRENLayer) = 0

# #-----------------------------------------------------------------------------# SIREN MLP

# """
#     build_siren_mlp(in_dims, hidden_dims, out_dims, n_layers; omega_0=30.0)

# Build a SIREN MLP with the specified architecture.

# # Arguments
# - `in_dims`: Input dimension
# - `hidden_dims`: Hidden layer dimension
# - `out_dims`: Output dimension
# - `n_layers`: Number of hidden layers
# - `omega_0`: Frequency parameter (default: 30.0)

# # Returns
# A Lux Chain representing the SIREN MLP
# """
# function build_siren_mlp(in_dims::Int, hidden_dims::Int, out_dims::Int, n_layers::Int; omega_0=30.0)
#     layers = []

#     # First layer
#     push!(layers, SIRENLayer(in_dims, hidden_dims; omega_0=omega_0, is_first=true))

#     # Hidden layers
#     for _ in 2:n_layers
#         push!(layers, SIRENLayer(hidden_dims, hidden_dims; omega_0=omega_0, is_first=false))
#     end

#     # Output layer (no activation)
#     push!(layers, Lux.Dense(hidden_dims, out_dims))

#     return Lux.Chain(layers...)
# end

# #-----------------------------------------------------------------------------# ImplicitTerrain Model

# """
#     ImplicitTerrain <: Lux.AbstractExplicitContainerLayer

# ImplicitTerrain model with two cascaded MLPs:
# - Surface Model (��): Predicts height from (x, y) coordinates
# - Geometry Model (��): Can be used for derived geometric properties

# Architecture (per paper):
# - 3 hidden layers with 256 units each
# - SIREN activation functions
# - Input: (x, y) normalized to [-1, 1]
# - Output: height z
# """
# struct ImplicitTerrain{S, M, C, E} <: Lux.AbstractLuxContainerLayer{(:surface_model, :geometry_model)}
#     surface_model::S
#     geometry_model::M
#     crs::C
#     extent::E
#     x_range::Tuple{Float32, Float32}
#     y_range::Tuple{Float32, Float32}
#     z_mean::Float32
#     z_std::Float32
# end

# """
#     ImplicitTerrain(; hidden_dims=256, n_layers=3, omega_0=30.0,
#                     crs=nothing, extent=nothing,
#                     x_range=(-1f0, 1f0), y_range=(-1f0, 1f0),
#                     z_mean=0f0, z_std=1f0)

# Create an ImplicitTerrain model.

# # Arguments
# - `hidden_dims`: Hidden layer dimension (default: 256)
# - `n_layers`: Number of hidden layers (default: 3)
# - `omega_0`: SIREN frequency parameter (default: 30.0)
# - `crs`: Coordinate reference system
# - `extent`: Spatial extent
# - `x_range`: Range for x normalization
# - `y_range`: Range for y normalization
# - `z_mean`: Mean for z denormalization
# - `z_std`: Standard deviation for z denormalization
# """
# function ImplicitTerrain(;
#     hidden_dims::Int=256,
#     n_layers::Int=3,
#     omega_0=30.0,
#     crs=nothing,
#     extent=nothing,
#     x_range=(-1f0, 1f0),
#     y_range=(-1f0, 1f0),
#     z_mean=0f0,
#     z_std=1f0
# )
#     # Surface model: (x, y) -> z
#     surface_model = build_siren_mlp(2, hidden_dims, 1, n_layers; omega_0=omega_0)

#     # Geometry model: Can be used for normals, curvature, etc.
#     # For now, same architecture as surface model
#     geometry_model = build_siren_mlp(2, hidden_dims, 3, n_layers; omega_0=omega_0)

#     return ImplicitTerrain(
#         surface_model, geometry_model, crs, extent,
#         x_range, y_range, z_mean, z_std
#     )
# end

# function (model::ImplicitTerrain)(xy, ps, st)
#     # Forward pass through surface model
#     z, st_surface = model.surface_model(xy, ps.surface_model, st.surface_model)

#     # Denormalize output
#     z_denorm = z .* model.z_std .+ model.z_mean

#     return z_denorm, (surface_model=st_surface, geometry_model=st.geometry_model)
# end

# """
#     predict_with_geometry(model::ImplicitTerrain, xy, ps, st)

# Predict both height and geometric properties (normals, etc.) from the geometry model.
# """
# function predict_with_geometry(model::ImplicitTerrain, xy, ps, st)
#     # Surface prediction
#     z, st_surface = model.surface_model(xy, ps.surface_model, st.surface_model)
#     z_denorm = z .* model.z_std .+ model.z_mean

#     # Geometry prediction (e.g., normals)
#     geom, st_geom = model.geometry_model(xy, ps.geometry_model, st.geometry_model)

#     return (height=z_denorm, geometry=geom), (surface_model=st_surface, geometry_model=st_geom)
# end

# #-----------------------------------------------------------------------------# Training utilities

# """
#     normalize_coordinates(x, y, x_range, y_range)

# Normalize coordinates to [-1, 1] range.
# """
# function normalize_coordinates(x, y, x_range, y_range)
#     x_norm = 2f0 .* (x .- x_range[1]) ./ (x_range[2] - x_range[1]) .- 1f0
#     y_norm = 2f0 .* (y .- y_range[1]) ./ (y_range[2] - y_range[1]) .- 1f0
#     return x_norm, y_norm
# end

# """
#     prepare_training_data(raster::Raster)

# Prepare training data from a Raster object.

# Returns:
# - `x_coords`: x coordinates
# - `y_coords`: y coordinates
# - `z_values`: height values
# - `x_range`: (min, max) for x
# - `y_range`: (min, max) for y
# - `z_mean`: mean of z values
# - `z_std`: standard deviation of z values
# """
# function prepare_training_data(raster)
#     # Extract coordinates and values
#     xy = xyfeatures(raster)
#     x_coords = Float32.(xy.x)
#     y_coords = Float32.(xy.y)
#     z_values = Float32.(vec(raster.data))

#     # Compute normalization parameters
#     x_range = (minimum(x_coords), maximum(x_coords))
#     y_range = (minimum(y_coords), maximum(y_coords))
#     z_mean = mean(z_values)
#     z_std = std(z_values)

#     return (
#         x=x_coords,
#         y=y_coords,
#         z=z_values,
#         x_range=x_range,
#         y_range=y_range,
#         z_mean=z_mean,
#         z_std=z_std
#     )
# end

# """
#     create_implicit_terrain_from_raster(raster::Raster; kwargs...)

# Create and return an ImplicitTerrain model initialized with data statistics from a raster.

# # Arguments
# - `raster`: Input Raster object
# - `kwargs...`: Additional arguments passed to ImplicitTerrain constructor

# # Returns
# - `model`: ImplicitTerrain model
# - `data`: Prepared training data
# """
# function create_implicit_terrain_from_raster(raster; kwargs...)
#     data = prepare_training_data(raster)

#     model = ImplicitTerrain(;
#         crs=GI.crs(raster),
#         extent=GI.extent(raster),
#         x_range=data.x_range,
#         y_range=data.y_range,
#         z_mean=data.z_mean,
#         z_std=data.z_std,
#         kwargs...
#     )

#     return model, data
# end

# #-----------------------------------------------------------------------------# Interface implementations

# # Make ImplicitTerrain compatible with AbstractGeoSurrogate interface
# GI.crs(model::ImplicitTerrain) = model.crs
# GI.extent(model::ImplicitTerrain) = model.extent

# """
#     (model::ImplicitTerrain)(x::Real, y::Real, ps, st)

# Predict height at a single coordinate point.
# """
# function (model::ImplicitTerrain)(x::Real, y::Real, ps, st)
#     x_norm, y_norm = normalize_coordinates([x], [y], model.x_range, model.y_range)
#     xy = vcat(x_norm', y_norm')
#     z, st_new = model(xy, ps, st)
#     return z[1], st_new
# end

# """
#     predict_height(model::ImplicitTerrain, x, y, ps, st)

# Predict height values for arrays of coordinates.
# """
# function predict_height(model::ImplicitTerrain, x, y, ps, st)
#     x_norm, y_norm = normalize_coordinates(x, y, model.x_range, model.y_range)
#     xy = vcat(reshape(x_norm, 1, :), reshape(y_norm, 1, :))
#     z, st_new = model(xy, ps, st)
#     return vec(z), st_new
# end


end  # module ImplicitTerrain

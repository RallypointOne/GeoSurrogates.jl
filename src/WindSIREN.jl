module WindSurrogate

using ADTypes, Lux, Random, Optimisers, StyledStrings, Rasters, DataFrames, Images

using ..GeoSurrogates: is_normalized, normalize, SIRENActivation, init_weight_first, init_weight_hidden, init_bias_zeros

import StatsAPI: predict, fit!
import DimensionalData as DD
import RapidRefreshData as RAP

#-----------------------------------------------------------------------------# WindSIREN
"""
    WindSIREN(; kwargs...)

A SIREN (Sinusoidal Representation Network) for creating implicit neural representations
of wind field (u, v) data. Uses periodic sine activations to capture high-frequency
spatial patterns in wind fields.

# Keyword Arguments
- `in::Int = 2`: Input dimension (typically 2 for x, y coordinates)
- `hidden::Int = 256`: Number of hidden units per layer
- `out::Int = 2`: Output dimension (2 for u, v wind components)
- `n_hidden::Int = 3`: Number of hidden layers
- `ω0::Float32 = 30f0`: Frequency scaling for first layer (controls initial frequency)
- `ωh::Float32 = 1f0`: Frequency scaling for hidden layers
- `rng::AbstractRNG = Random.MersenneTwister(42)`: Random number generator
- `alg = Adam(0.0001f0)`: Optimizer algorithm

# Example
```julia
model = WindSIREN()
fit!(model, u_raster, v_raster; steps=1000)
u_pred, v_pred = predict(model, test_raster)
```

# References
- Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions" (2020)
"""
struct WindSIREN{T, P, S, TS}
    chain::T
    parameters::P
    states::S
    train_state::TS

    function WindSIREN(;
                in = 2,
                hidden = 256,
                out = 2,  # u and v components
                n_hidden = 3,
                ω0 = 30f0,
                ωh = 1f0,
                rng = Random.MersenneTwister(42),
                alg = Adam(0.0001f0),
            )
        # Build SIREN layers using SIRENActivation functor (avoids closure overhead)
        layers = [
            # First layer with ω0 scaling
            Dense(in, hidden, SIRENActivation(ω0); init_weight = init_weight_first, init_bias = init_bias_zeros),
            # Hidden layers with ωh scaling (typically ωh=1)
            [Dense(hidden, hidden, SIRENActivation(ωh); init_weight = init_weight_hidden, init_bias = init_bias_zeros) for _ in 2:n_hidden]...,
            # Output layer (linear, no activation for regression)
            Dense(hidden, out; init_weight = init_weight_hidden, init_bias = init_bias_zeros)
        ]
        model = Chain(layers...)
        ps, st = Lux.setup(rng, model)
        ts = Lux.Training.TrainState(model, ps, st, alg)
        new{typeof(model), typeof(ps), typeof(st), typeof(ts)}(model, ps, st, ts)
    end
end

Base.show(io::IO, ::MIME"text/plain", o::WindSIREN) = print(io, "WindSurrogate.WindSIREN")
Base.show(io::IO, o::WindSIREN) = print(io, "WindSIREN")

#-----------------------------------------------------------------------------# predict
"""
    predict(model::WindSIREN, x::AbstractMatrix) -> Matrix{Float32}

Predict wind components (u, v) for input coordinates.
Input `x` should be a 2×N matrix of normalized coordinates.
Returns a 2×N matrix where row 1 is u and row 2 is v.
"""
function predict(o::WindSIREN, x::AbstractMatrix)
    Lux.apply(o.chain, x, o.train_state.parameters, o.train_state.states)[1]
end

"""
    predict(model::WindSIREN, coords::Tuple) -> Tuple{Float32, Float32}

Predict wind components (u, v) for a single coordinate tuple (x, y).
Returns (u, v) tuple.
"""
function predict(o::WindSIREN, coords::Tuple)
    out = predict(o, hcat(collect(Float32, coords)))
    return (out[1], out[2])
end

"""
    predict(model::WindSIREN, r::Raster) -> Tuple{Raster, Raster}

Predict wind components for all coordinates in a raster.
Returns tuple of (u_raster, v_raster).
"""
function predict(o::WindSIREN, r::Raster)
    data = features(r)
    ŷ = predict(o, data)
    u_raster = Raster(reshape(ŷ[1, :], size(r)), dims=DD.dims(r))
    v_raster = Raster(reshape(ŷ[2, :], size(r)), dims=DD.dims(r))
    return (u_raster, v_raster)
end

#-----------------------------------------------------------------------------# features
"""
    features(r::Raster) -> Matrix{Float32}

Extract normalized (x, y) coordinate features from a raster.
Returns a 2×N matrix of normalized coordinates.
"""
function features(r::Raster)
    df = DataFrame(r)
    x = normalize(df.X)
    y = normalize(df.Y)
    return Float32[x'; y']
end

#-----------------------------------------------------------------------------# fit!
"""
    fit!(model::WindSIREN, x::AbstractMatrix, y::AbstractMatrix; steps=1)

Train the model on coordinate-value pairs.
- `x`: 2×N matrix of normalized (x, y) coordinates
- `y`: 2×N matrix of (u, v) wind components

Returns the fitted model.
"""
function fit!(o::WindSIREN, x::AbstractMatrix, y::AbstractMatrix; steps = 1)
    for _ in 1:steps
        Lux.Training.single_train_step!(AutoZygote(), Lux.MSELoss(), (x, y), o.train_state)
    end
    return o
end

"""
    fit!(model::WindSIREN, u::Raster, v::Raster; steps=1)

Train the model on u and v wind component rasters.
Rasters must be normalized before fitting.

Returns the fitted model.
"""
function fit!(o::WindSIREN, u::Raster, v::Raster; steps = 1)
    is_normalized(u) || error("U raster must be normalized (see GeoSurrogates.normalize) before fitting.")
    is_normalized(v) || error("V raster must be normalized (see GeoSurrogates.normalize) before fitting.")
    size(u) == size(v) || error("U and V rasters must have the same size.")

    x = features(u)  # Use u raster for coordinates (both should have same grid)
    y = Float32[vec(u.data)'; vec(v.data)']
    return fit!(o, x, y; steps=steps)
end

"""
    fit!(model::WindSIREN, uv::RasterStack; steps=1)

Train the model on a RasterStack containing :u and :v layers.

Returns the fitted model.
"""
function fit!(o::WindSIREN, uv::DD.AbstractDimStack; steps = 1)
    haskey(uv, :u) || error("RasterStack must have a :u layer")
    haskey(uv, :v) || error("RasterStack must have a :v layer")
    return fit!(o, uv[:u], uv[:v]; steps=steps)
end


end

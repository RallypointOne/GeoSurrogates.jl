module CategoricalSIREN

using ADTypes, Lux, Random, Optimisers, Rasters, DataFrames

using ..GeoSurrogates: normalize, SIRENActivation, init_weight_first, init_weight_hidden, init_bias_zeros

import StatsAPI: predict, fit!
import DimensionalData as DD

#-----------------------------------------------------------------------------# CategoricalSIREN
"""
    CatSIREN(n_classes; kwargs...)

A SIREN (Sinusoidal Representation Network) for creating implicit neural representations
of categorical raster data (e.g., Landfire fuel models). Uses periodic sine activations
with a softmax output layer to ensure predictions sum to 1.

# Arguments
- `n_classes::Int`: Number of output classes (required)

# Keyword Arguments
- `in::Int = 2`: Input dimension (typically 2 for x, y coordinates)
- `hidden::Int = 256`: Number of hidden units per layer
- `n_hidden::Int = 3`: Number of hidden layers
- `ω0::Float32 = 30f0`: Frequency scaling for first layer
- `ωh::Float32 = 1f0`: Frequency scaling for hidden layers
- `rng::AbstractRNG = Random.MersenneTwister(42)`: Random number generator
- `alg = Adam(0.0001f0)`: Optimizer algorithm

# Example
```julia
# For a categorical raster with 10 unique classes
model = CatSIREN(10)
fit!(model, categorical_raster; steps=1000)
probs = predict(model, test_raster)  # Returns raster of probability vectors
```

# References
- Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions" (2020)
"""
struct CatSIREN{T, P, S, TS, C}
    chain::T
    parameters::P
    states::S
    train_state::TS
    classes::C  # Store the class labels for mapping back

    function CatSIREN(n_classes::Int;
                in = 2,
                hidden = 256,
                n_hidden = 3,
                ω0 = 30f0,
                ωh = 1f0,
                rng = Random.MersenneTwister(42),
                alg = Adam(0.0001f0),
                classes = nothing,
            )
        # Build SIREN layers with softmax output using SIRENActivation functor
        layers = [
            # First layer with ω0 scaling
            Dense(in, hidden, SIRENActivation(ω0); init_weight = init_weight_first, init_bias = init_bias_zeros),
            # Hidden layers with ωh scaling
            [Dense(hidden, hidden, SIRENActivation(ωh); init_weight = init_weight_hidden, init_bias = init_bias_zeros) for _ in 2:n_hidden]...,
            # Output layer with softmax for probability distribution
            Dense(hidden, n_classes; init_weight = init_weight_hidden, init_bias = init_bias_zeros),
            softmax,
        ]
        model = Chain(layers...)
        ps, st = Lux.setup(rng, model)
        ts = Lux.Training.TrainState(model, ps, st, alg)
        new{typeof(model), typeof(ps), typeof(st), typeof(ts), typeof(classes)}(model, ps, st, ts, classes)
    end
end

Base.show(io::IO, ::MIME"text/plain", o::CatSIREN) = print(io, "CategoricalSIREN.CatSIREN(n_classes=$(length(o.classes)))")
Base.show(io::IO, o::CatSIREN) = print(io, "CatSIREN($(length(o.classes)))")

#-----------------------------------------------------------------------------# predict
"""
    predict(model::CatSIREN, x::AbstractMatrix) -> Matrix{Float32}

Predict class probabilities for input coordinates.
Input `x` should be a 2×N matrix of normalized coordinates.
Returns an n_classes×N matrix of probabilities (columns sum to 1).
"""
function predict(o::CatSIREN, x::AbstractMatrix)
    Lux.apply(o.chain, x, o.train_state.parameters, o.train_state.states)[1]
end

"""
    predict(model::CatSIREN, coords::Tuple) -> Vector{Float32}

Predict class probabilities for a single coordinate tuple (x, y).
Returns a vector of probabilities that sum to 1.
"""
function predict(o::CatSIREN, coords::Tuple)
    out = predict(o, hcat(collect(Float32, coords)))
    return vec(out)
end

"""
    predict(model::CatSIREN, r::Raster) -> Raster

Predict class probabilities for all coordinates in a raster.
Returns a Raster where each cell contains a Dict mapping class labels to probabilities.
"""
function predict(o::CatSIREN, r::Raster)
    data = features(r)
    ŷ = predict(o, data)  # n_classes × N matrix
    # Convert to vector of Dicts
    result = [Dict(o.classes[i] => ŷ[i, j] for i in eachindex(o.classes)) for j in axes(ŷ, 2)]
    return Raster(reshape(result, size(r)), dims=DD.dims(r))
end

"""
    predict_class(model::CatSIREN, x) -> class_label

Return the most likely class for the given input.
"""
function predict_class(o::CatSIREN, x::AbstractMatrix)
    probs = predict(o, x)
    indices = [argmax(probs[:, j]) for j in axes(probs, 2)]
    return [o.classes[i] for i in indices]
end

function predict_class(o::CatSIREN, coords::Tuple)
    probs = predict(o, coords)
    return o.classes[argmax(probs)]
end

function predict_class(o::CatSIREN, r::Raster)
    data = features(r)
    classes = predict_class(o, data)
    return Raster(reshape(classes, size(r)), dims=DD.dims(r))
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

#-----------------------------------------------------------------------------# one-hot encoding
"""
    onehot(labels, classes) -> Matrix{Float32}

Create one-hot encoded matrix from labels.
Returns an n_classes × N matrix.
"""
function onehot(labels::AbstractVector, classes::AbstractVector)
    n_classes = length(classes)
    n_samples = length(labels)
    class_to_idx = Dict(c => i for (i, c) in enumerate(classes))
    result = zeros(Float32, n_classes, n_samples)
    for (j, label) in enumerate(labels)
        if !ismissing(label) && haskey(class_to_idx, label)
            result[class_to_idx[label], j] = 1f0
        end
    end
    return result
end

#-----------------------------------------------------------------------------# Cross-entropy loss
"""
    CrossEntropyLoss()

Cross-entropy loss function compatible with Lux's training API.
"""
struct CrossEntropyLoss end

function (::CrossEntropyLoss)(model, ps, st, (x, y))
    ŷ, st_new = Lux.apply(model, x, ps, st)
    # Add small epsilon to avoid log(0)
    ε = 1f-7
    loss = -sum(y .* log.(ŷ .+ ε)) / size(y, 2)
    return loss, st_new, (;)
end

#-----------------------------------------------------------------------------# fit!
"""
    fit!(model::CatSIREN, x::AbstractMatrix, y::AbstractMatrix; steps=1)

Train the model on coordinate-value pairs.
- `x`: 2×N matrix of normalized (x, y) coordinates
- `y`: n_classes×N one-hot encoded matrix

Returns the fitted model.
"""
function fit!(o::CatSIREN, x::AbstractMatrix, y::AbstractMatrix; steps = 1)
    loss_fn = CrossEntropyLoss()
    for _ in 1:steps
        Lux.Training.single_train_step!(AutoZygote(), loss_fn, (x, y), o.train_state)
    end
    return o
end

"""
    fit!(model::CatSIREN, r::Raster; steps=1)

Train the model on a categorical raster.
The raster values should be categorical labels (integers or any comparable type).

Returns the fitted model.
"""
function fit!(o::CatSIREN, r::Raster; steps = 1)
    x = features(r)
    labels = vec(r.data)
    y = onehot(labels, o.classes)
    return fit!(o, x, y; steps=steps)
end

#-----------------------------------------------------------------------------# Constructor from Raster
"""
    CatSIREN(r::Raster; kwargs...)

Create a CatSIREN model from a categorical raster, automatically determining
the number of classes from the unique values in the raster.
"""
function CatSIREN(r::Raster; kwargs...)
    classes = sort(unique(skipmissing(r.data)))
    n_classes = length(classes)
    CatSIREN(n_classes; classes=classes, kwargs...)
end

end

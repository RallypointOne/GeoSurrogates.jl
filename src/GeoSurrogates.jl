module GeoSurrogates

using ADTypes, ArchGDAL, CategoricalArrays, DataFrames, Extents, Interpolations, Images, LazyArrays, LinearAlgebra, Lux, NNlib, Optimisers, Random, Rasters, RasterDataSources, Statistics, StatsAPI, StatsModels, StyledStrings, Tables

import Extents
import GeoInterface as GI
import GeometryOps as GO
import DimensionalData as DD

import StatsAPI: predict, fit!

export ImplicitTerrain, WindSurrogate, CatSIREN, predict, fit!
export LinReg, IDW, RBF, TPS, RasterWrap, CategoricalRasterWrap, GeomWrap, AdditiveModel, normalize, gaussian


#-----------------------------------------------------------------------------# normalize
# *normalize* in this package means scaling to [-1, 1]
function normalize(x::AbstractArray{<:Union{Missing, Number}})
    vals = skipmissing(x)
    isempty(vals) && return x
    a, b = extrema(vals)
    a == b && return zero(x)  # All values equal → map to 0 (midpoint of [-1, 1])
    return @. 2 * (x - a) / (b - a) - 1
end

# Fallback does nothing (e.g. for strings)
normalize(x) = x

normalize(dim::DD.Dimension) = DD.Lookups.iscategorical(dim) ? dim : DD.rebuild(dim, normalize(dim.val.data))

function normalize(r::Raster)
    normalized_dims = map(normalize, DD.dims(r))
    normalized_data = normalize(r.data)
    return Raster(normalized_data; dims=normalized_dims)
end

normalize(nt::NamedTuple) = map(normalize, nt)

is_normalized(x) = all(v -> -1 ≤ v ≤ 1, skipmissing(x))

function normalize!(df::DataFrame)
    for col in names(df)
        df[!, col] = normalize(df[!, col])
    end
    return df
end

#-----------------------------------------------------------------------------# GeoSurrogate
abstract type GeoSurrogate end

Base.show(io::IO, o::T) where {T <: GeoSurrogate} = print(io, T.name.name)

# Interface:
GI.crs(::GeoSurrogate) = nothing
GI.extent(::GeoSurrogate) = nothing

predict(model::GeoSurrogate, x::Tuple) = error("predict not implemented for $(typeof(model)) with input type Tuple")
predict(model::GeoSurrogate, x::Raster) = error("predict not implemented for $(typeof(model)) with input type Raster")


#-----------------------------------------------------------------------------# AdditiveModel
"""
    AdditiveModel(models::Vector)

A boosting-style composite surrogate.  Each model is fitted to the residuals (target minus the
sum of all previous model predictions), and the final prediction is the sum of all model predictions.

### Examples

```julia
model = AdditiveModel([ImplicitTerrain.MLP(), ImplicitTerrain.MLP()])
fit!(model, raster; steps=1000)
predict(model, raster)
```
"""
struct AdditiveModel{T} <: GeoSurrogate
    models::Vector{T}
end

function fit!(o::AdditiveModel, r::Raster; kw...)
    residual = r
    for (i, model) in enumerate(o.models)
        fit!(model, residual; kw...)
        if i < length(o.models)
            residual = Raster(Matrix(r) .- sum(Matrix(predict(o.models[j], r)) for j in 1:i), dims=DD.dims(r))
        end
    end
    return o
end

predict(o::AdditiveModel, x::Tuple) = sum(predict(m, x) for m in o.models)

function predict(o::AdditiveModel, r::Raster)
    predictions = [Matrix(predict(m, r)) for m in o.models]
    Raster(sum(predictions), dims=DD.dims(r))
end

#-----------------------------------------------------------------------------# LinReg
struct LinReg{F <: FormulaTerm, T <: Number} <: GeoSurrogate
    formula::F
    β::Vector{T}
end
Base.show(io::IO, o::LinReg) = print(io, "LinReg: ", o.formula)

function LinReg(r::Raster, formula = @formula(layer1 ~ 1 + X * Y ))
    df = dropmissing!(DataFrame(r))
    f = apply_schema(formula, schema(formula, df))
    y, x = modelcols(f, df)
    β = x \ y
    return LinReg(f, β)
end

# No-op: LinReg is fitted at construction
fit!(o::LinReg, ::Raster; kw...) = o

function predict(o::LinReg, coords::Tuple)
    nt = (X = coords[1], Y = coords[2])
    only(modelmatrix(o.formula, Tables.rowtable((nt, ))) * o.β)
end

function predict(o::LinReg, r::Raster)
    df = DataFrame(r)
    X = modelmatrix(o.formula, df)
    ẑ = X * o.β
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

#-----------------------------------------------------------------------------# IDW (Inverse Distance Weighting)
"""
    IDW(r::Raster; power=2)

Inverse Distance Weighting surrogate.  Predicts values as a weighted average of known data
points, where weights are inversely proportional to the distance raised to `power`.

### Examples

```julia
model = IDW(raster)
predict(model, (x, y))
predict(model, raster)
```
"""
struct IDW{T <: AbstractFloat} <: GeoSurrogate
    xs::Vector{T}
    ys::Vector{T}
    zs::Vector{T}
    power::T
end

function IDW(r::Raster; power=2)
    df = dropmissing!(DataFrame(r))
    T = promote_type(eltype(df.X), eltype(df.Y), eltype(df.layer1), typeof(float(power)))
    IDW(T.(df.X), T.(df.Y), T.(df.layer1), T(power))
end

fit!(o::IDW, ::Raster; kw...) = o

function predict(o::IDW, coords::Tuple)
    x, y = coords
    T = eltype(o.xs)
    xf, yf = T(x), T(y)
    p = o.power
    num = zero(T)
    den = zero(T)
    @inbounds for i in eachindex(o.xs, o.ys, o.zs)
        dx = xf - o.xs[i]
        dy = yf - o.ys[i]
        d2 = dx * dx + dy * dy
        d2 == zero(T) && return o.zs[i]
        w = inv(d2^(p / 2))
        num += w * o.zs[i]
        den += w
    end
    return num / den
end

function predict(o::IDW, r::Raster)
    ẑ = [predict(o, pt) for pt in DimPoints(r)]
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

#-----------------------------------------------------------------------------# RBF (Radial Basis Function)
rbf_gaussian(r, ε) = exp(-(ε * r)^2)
rbf_multiquadric(r, ε) = sqrt(1 + (ε * r)^2)
rbf_inverse_multiquadric(r, ε) = inv(sqrt(1 + (ε * r)^2))
rbf_linear(r, ε) = r
rbf_cubic(r, ε) = r^3
rbf_thin_plate_spline(r, ε) = r > 0 ? r^2 * log(r) : zero(r)

const RBF_KERNELS = Dict{Symbol, Function}(
    :gaussian              => rbf_gaussian,
    :multiquadric          => rbf_multiquadric,
    :inverse_multiquadric  => rbf_inverse_multiquadric,
    :linear                => rbf_linear,
    :cubic                 => rbf_cubic,
    :thin_plate_spline     => rbf_thin_plate_spline,
)

"""
    RBF(r::Raster; kernel=:gaussian, epsilon=1.0, poly_degree=0)

Radial Basis Function surrogate.  Interpolates scattered data using a kernel applied to
pairwise distances.  Set `poly_degree=1` to augment with a linear polynomial term.

Available kernels: `:gaussian`, `:multiquadric`, `:inverse_multiquadric`, `:linear`,
`:cubic`, `:thin_plate_spline`.

### Examples

```julia
model = RBF(raster; kernel=:gaussian, epsilon=1.0)
predict(model, (x, y))
```
"""
struct RBF{T <: AbstractFloat, K} <: GeoSurrogate
    xs::Vector{T}
    ys::Vector{T}
    weights::Vector{T}
    kernel::K
    epsilon::T
    poly_degree::Int
end

function RBF(r::Raster; kernel=:gaussian, epsilon=1.0, poly_degree=0)
    df = dropmissing!(DataFrame(r))
    T = promote_type(eltype(df.X), eltype(df.Y), eltype(df.layer1), typeof(float(epsilon)))
    xs = T.(df.X)
    ys = T.(df.Y)
    zs = T.(df.layer1)
    n = length(xs)
    ε = T(epsilon)
    kfn = kernel isa Symbol ? RBF_KERNELS[kernel] : kernel

    # Build distance matrix and apply kernel
    Φ = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        d = hypot(xs[i] - xs[j], ys[i] - ys[j])
        Φ[i, j] = kfn(d, ε)
    end

    if poly_degree >= 1
        P = hcat(ones(T, n), xs, ys)
        Z = zeros(T, 3, 3)
        A = [Φ P; P' Z]
        rhs = [zs; zeros(T, 3)]
        w = A \ rhs
    else
        w = Φ \ zs
    end

    RBF(xs, ys, w, kfn, ε, poly_degree)
end

fit!(o::RBF, ::Raster; kw...) = o

function predict(o::RBF, coords::Tuple)
    x, y = coords
    T = eltype(o.xs)
    xf, yf = T(x), T(y)
    n = length(o.xs)
    val = zero(T)
    @inbounds for i in 1:n
        d = hypot(xf - o.xs[i], yf - o.ys[i])
        val += o.weights[i] * o.kernel(d, o.epsilon)
    end
    if o.poly_degree >= 1
        val += o.weights[n+1] + o.weights[n+2] * xf + o.weights[n+3] * yf
    end
    return val
end

function predict(o::RBF, r::Raster)
    ẑ = [predict(o, pt) for pt in DimPoints(r)]
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

#-----------------------------------------------------------------------------# TPS (Thin Plate Spline)
_tps_kernel(r) = r > 0 ? r^2 * log(r) : zero(r)

"""
    TPS(r::Raster; regularization=0.0)

Thin Plate Spline surrogate.  Always includes an affine term (`a₀ + a₁x + a₂y`).
Set `regularization > 0` for smoothing instead of exact interpolation.

### Examples

```julia
model = TPS(raster)
predict(model, (x, y))
```
"""
struct TPS{T <: AbstractFloat} <: GeoSurrogate
    xs::Vector{T}
    ys::Vector{T}
    weights::Vector{T}
    affine::Vector{T}
    regularization::T
end

function TPS(r::Raster; regularization=0.0)
    df = dropmissing!(DataFrame(r))
    T = promote_type(eltype(df.X), eltype(df.Y), eltype(df.layer1), typeof(float(regularization)))
    xs = T.(df.X)
    ys = T.(df.Y)
    zs = T.(df.layer1)
    n = length(xs)
    λ = T(regularization)

    # Build kernel matrix
    Φ = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        d = hypot(xs[i] - xs[j], ys[i] - ys[j])
        Φ[i, j] = _tps_kernel(d)
    end

    P = hcat(ones(T, n), xs, ys)
    Z = zeros(T, 3, 3)
    A = [Φ + λ * I P; P' Z]
    rhs = [zs; zeros(T, 3)]
    w = A \ rhs

    TPS(xs, ys, w[1:n], w[n+1:n+3], λ)
end

fit!(o::TPS, ::Raster; kw...) = o

function predict(o::TPS, coords::Tuple)
    x, y = coords
    T = eltype(o.xs)
    xf, yf = T(x), T(y)
    val = o.affine[1] + o.affine[2] * xf + o.affine[3] * yf
    @inbounds for i in eachindex(o.xs, o.ys, o.weights)
        d = hypot(xf - o.xs[i], yf - o.ys[i])
        val += o.weights[i] * _tps_kernel(d)
    end
    return val
end

function predict(o::TPS, r::Raster)
    ẑ = [predict(o, pt) for pt in DimPoints(r)]
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end


#-----------------------------------------------------------------------------# RasterWrap
"""
    RasterWrap(r::Raster; int = BSpline(Linear()), ext = nothing)

A GeoSurrogate with interpolation (and optional extrapolation).
"""
struct RasterWrap{R <: Raster, A, E, F} <: GeoSurrogate
    raster::R
    interpolations_alg::A
    extrapolation_alg::E
    f::F
    function RasterWrap(r::Raster, int = BSpline(Linear()), ext = nothing)
        r = forwardorder(r)
        itp = interpolate(r.data, int)
        itps = scale(itp, (x.val.data for x in r.dims)...)
        f = isnothing(ext) ? itps : extrapolate(itps, ext)
        return new{typeof(r), typeof(int), typeof(ext), typeof(f)}(r, int, ext, f)
    end
end

predict(rw::RasterWrap, coords::Tuple) = rw.f(coords...)

function predict(rw::RasterWrap, r::Raster)
    r2 = forwardorder(r)
    xs = r2.dims[1].val.data
    ys = r2.dims[2].val.data
    ẑ = [rw.f(x, y) for x in xs, y in ys]
    return Raster(ẑ, dims=DD.dims(r2))
end

GI.crs(o::RasterWrap) = GI.crs(o.raster)
GI.extent(o::RasterWrap) = GI.extent(o.raster)

_dimstep(d) = _dimstep(d.val.data)
_dimstep(r::AbstractRange) = step(r)
_dimstep(v::AbstractVector) = length(v) < 2 ? zero(eltype(v)) : v[2] - v[1]

function forwardorder(r::Raster)
    for (i, dim) in enumerate(r.dims)
        if _dimstep(dim) < 0
            r = reverse(r, dims=i)
        end
    end
    return r
end

#-----------------------------------------------------------------------------# CategoricalRasterWrap
"""
    CategoricalRasterWrap(r::Raster; kernel)

A GeoSurrogate for categorical rasters using kernel smoothing.  Here, "categorical raster" refers
to a raster where its numerical data is associated with discrete categories (e.g. land cover types).
"""
struct CategoricalRasterWrap{R, T, G, K} <: GeoSurrogate
    raster::R
    geomwraps::Dict{T, G}  # Dict{eltype(raster), GeomWrap} - cached GeomWrap per class
    kernel::K
end
function CategoricalRasterWrap(r::Raster; kernel = Base.Fix2(gaussian, 4))
    df = DataFrame(r)
    gdf = groupby(df, :layer1)
    # Pre-create GeomWrap objects for each class to avoid repeated allocation
    geomwraps = Dict(
        k.layer1 => GeomWrap(GI.MultiPoint([(row.X, row.Y) for row in eachrow(group)]); kernel)
        for (k, group) in pairs(gdf)
    )
    CategoricalRasterWrap(r, geomwraps, kernel)
end

GI.crs(o::CategoricalRasterWrap) = GI.crs(o.raster)
GI.extent(o::CategoricalRasterWrap) = GI.extent(o.raster)

function predict(crw::CategoricalRasterWrap, coords::Tuple)
    keys_vec = collect(keys(crw.geomwraps))
    vals = [predict(crw.geomwraps[k], coords) for k in keys_vec]
    s = sum(vals)
    if s != 0
        vals ./= s
    end
    return Dict(keys_vec[i] => vals[i] for i in eachindex(keys_vec))
end

function predict(crw::CategoricalRasterWrap, r::Raster)
    keys_vec = collect(keys(crw.geomwraps))
    pts = collect(DimPoints(r))
    n = length(pts)
    nk = length(keys_vec)
    # Pre-allocate result matrix: n_classes × n_points
    result = Matrix{Float64}(undef, nk, n)
    for (ci, k) in enumerate(keys_vec)
        gw = crw.geomwraps[k]
        for (j, pt) in enumerate(pts)
            result[ci, j] = predict(gw, pt)
        end
    end
    # Normalize columns to sum to 1
    for j in 1:n
        s = sum(@view result[:, j])
        if s != 0
            @view(result[:, j]) ./= s
        end
    end
    ẑ = [Dict(keys_vec[i] => result[i, j] for i in eachindex(keys_vec)) for j in 1:n]
    return Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

#-----------------------------------------------------------------------------# GeomWrap
"""
    GeomWrap(geometry; kernel)

A GeoSurrogate representing a geometry.  Coordinates contained in the geometry will result in
`predict(::GeomWrap, coords)) == 1.0`.  Coordinates "far" will be `0.0`.  "Near" coordinates are
determined by the `kernel` function evaluated at the distance between the coordinate and the geometry.
"""
struct GeomWrap{G, K} <: GeoSurrogate
    geometry::G
    kernel::K
    GeomWrap(geom::G; kernel::K = Base.Fix2(gaussian, 4)) where {G, K} = new{G,K}(geom, kernel)
end

gaussian(u, k) = exp(-0.5 * (k * u) * (k * u))

#-----------------------------------------------------------------------------# SIREN utilities (shared across modules)
# SIREN activation functor (avoids closure capture overhead)
struct SIRENActivation{T} <: Function
    ω::T
end
(s::SIRENActivation)(x) = sin(s.ω * x)

# NNlib compatibility for fast activation
import NNlib: fast_act
fast_act(s::SIRENActivation, ::AbstractArray) = s

# Weight initialization for SIREN networks
# First layer initialization: uniform in [-1/in, 1/in]
init_weight_first(rng, out, in) = (2rand(rng, Float32, out, in) .- 1) ./ in

# Hidden layer initialization: uniform in [-sqrt(6/in), sqrt(6/in)]
function init_weight_hidden(rng, out, in)
    limit = sqrt(6f0 / in)
    return limit .* (2rand(rng, Float32, out, in) .- 1)
end

init_bias_zeros(rng, out) = zeros(Float32, out)

GI.crs(o::GeomWrap) = GI.crs(o.geometry)
GI.extent(o::GeomWrap) = GI.extent(o.geometry)

function predict(o::GeomWrap, coords::Tuple)
    return o.kernel(GO.distance(GI.Point(coords), o.geometry))
end

function predict(o::GeomWrap, r::Raster)
    geom = o.geometry
    k = o.kernel
    ẑ = [k(GO.distance(GI.Point(pt), geom)) for pt in DimPoints(r)]
    return Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

#-----------------------------------------------------------------------------# ImplicitTerrain
include("ImplicitTerrain.jl")

include("WindSIREN.jl")

include("CatSIREN.jl")

end

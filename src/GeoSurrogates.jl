module GeoSurrogates

using ADTypes, ArchGDAL, CategoricalArrays, DataFrames, Extents, Interpolations, Images, LazyArrays, Lux, Optimisers, Random, Rasters, RasterDataSources, Statistics, StatsAPI, StatsModels, StyledStrings, Tables

import Extents
import GeoInterface as GI
import GeometryOps as GO
import DimensionalData as DD

import StatsAPI: predict, fit!

export ImplicitTerrain, WindSurrogate, predict, fit!


#-----------------------------------------------------------------------------# normalize
# *normalize* in this package means scaling to [-1, 1]
function normalize(x::AbstractArray{<:Union{Missing, Number}})
    vals = skipmissing(x)
    isempty(vals) && return x
    a, b = extrema(vals)
    a == b && return zero(x)  # All values equal → map to 0 (midpoint of [-1, 1])
    return 2 * (x .- a) ./ (b - a) .- 1
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

function predict(o::LinReg, coords::Tuple)
    nt = (X = coords[1], Y = coords[2])
    only(modelmatrix(o.formula, Tables.rowtable((nt, ))) * o.β)
end

function predict(o::LinReg, r::Raster)
    df = DataFrame(r)
    X = modelmatrix(o.formula, df)
    ẑ = X * o.β
    Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
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
predict(rw::RasterWrap, r::Raster) = nothing

GI.crs(o::RasterWrap) = GI.crs(o.raster)
GI.extent(o::RasterWrap) = GI.extent(o.raster)

function forwardorder(r::Raster)
    for (i, dim) in enumerate(r.dims)
        if step(dim.val.data) < 0
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
struct CategoricalRasterWrap{R, T, MP, K} <: GeoSurrogate
    raster::R
    dict::Dict{T, MP}  # Dict{eltype(raster), Vector{Tuple{Float64, Float64}}}
    kernel::K
end
function CategoricalRasterWrap(r::Raster; kernel = Base.Fix2(gaussian, 4))
    df = DataFrame(r)
    gdf = groupby(df, :layer1)
    dict = Dict(
        k.layer1 => [(row.X, row.Y) for row in eachrow(group)]
        for (k, group) in pairs(gdf)
    )
    CategoricalRasterWrap(r, dict, kernel)
end

GI.crs(o::CategoricalRasterWrap) = GI.crs(o.raster)
GI.extent(o::CategoricalRasterWrap) = GI.extent(o.raster)

function predict(crw::CategoricalRasterWrap, coords::Tuple)
    out = Dict(k => predict(GeomWrap(GI.MultiPoint(points)), coords) for (k, points) in crw.dict)
    sumvals = sum(values(out))
    sumvals == 0 ? out : Dict(k => v / sumvals for (k, v) in out)
end

function predict(crw::CategoricalRasterWrap, r::Raster)
    ẑ = [predict(crw, pt) for pt in DimPoints(r)]
    return Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
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

gaussian(u, k) = exp(-0.5 * (k * u) ^ 2)

GI.crs(o::GeomWrap) = GI.crs(o.geometry)
GI.extent(o::GeomWrap) = GI.extent(o.geometry)

predict(o::GeomWrap, coords::Tuple) = o.kernel(GO.distance(GI.Point(coords), o.geometry))

function predict(o::GeomWrap, r::Raster)
    ẑ = [predict(o, pt) for pt in DimPoints(r)]
    return Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
end

#-----------------------------------------------------------------------------# ImplicitTerrain
include("ImplicitTerrain.jl")

include("WindSurrogate.jl")

end

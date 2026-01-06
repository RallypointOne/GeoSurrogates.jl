module GeoSurrogates

using ADTypes, ArchGDAL, CategoricalArrays, DataFrames, Extents, Interpolations, Images, LazyArrays, Lux, Optimisers, Random, Rasters, RasterDataSources, Statistics, StatsAPI, StatsModels, StyledStrings, Tables

import Extents
import GeoInterface as GI
import GeometryOps as GO
import DimensionalData as DD

import StatsAPI: predict, fit!

export ImplicitTerrain


#-----------------------------------------------------------------------------# normalize
# *normalize* in this package means scaling to [-1, 1]
normalize(x::AbstractArray{<:Number}, a = minimum(x), b = maximum(x)) = 2 * (x .- a) ./ (b - a) .- 1

# Fallback does nothing (e.g. for strings)
normalize(x) = x

normalize(dim::DD.Dimension) = DD.Lookups.iscategorical(dim) ? dim : X(normalize(dim.val.data))

normalize(r::Raster) = modify(normalize, r)

normalize(nt::NamedTuple) = map(normalize, nt)

is_normalized(x) = all(-1 .≤ x .≤ 1)

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


# #-----------------------------------------------------------------------------# LinReg
# struct LinReg{F <: FormulaTerm, T <: Number} <: GeoSurrogate
#     formula::F
#     β::Vector{T}
# end

# function LinReg(r::Raster)
#     nt = merge(features(r), (; data = data(r)))
#     formula = fullfactorialformula(r)
#     f = apply_schema(formula, schema(formula, nt))
#     y, x = modelcols(f, nt)
#     β = x \ y
#     LinReg(f, β)
# end

# function (o::LinReg)(r::Raster)
#     X = modelmatrix(o.formula, features(r))
#     ẑ = X * o.β
#     Raster(reshape(ẑ, size(r)), dims=DD.dims(r))
# end

# function (o::LinReg)(x)
#     if Tables.istable(x)
#         X = modelmatrix(o.formula, x)
#         return X * o.β
#     else  # assume single row
#         only(o(Tables.rowtable((x, ))))
#     end
# end


#-----------------------------------------------------------------------------# RasterWrap
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


#-----------------------------------------------------------------------------# GeomWrap
struct GeomWrap{G, K} <: GeoSurrogate
    geometry::G
    kernel::K
    GeomWrap(geom::G, kern::K = Base.Fix2(gaussian, 4)) where {G, K} = new{G,K}(geom,kern)
end

(o::GeomWrap)(x...) = o.kernel(GO.distance(x, o.geometry))

gaussian(u, k) = exp(-0.5 * (k * u) ^ 2)

GI.crs(o::GeomWrap) = GI.crs(o.geometry)
GI.extent(o::GeomWrap) = GI.extent(o.geometry)

#-----------------------------------------------------------------------------# ImplicitTerrain
include("ImplicitTerrain.jl")

end

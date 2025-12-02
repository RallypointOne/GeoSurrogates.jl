module GeoSurrogates

using Extents, Rasters, Interpolations

import GeoInterface as GI
import GeometryOps as GO

#-----------------------------------------------------------------------------# features
function features(r::Raster)
    itr = Iterators.product(dims(r)...)
    out = Dict{Symbol, Vector{Union{eltype.(dims(r))...}}}()
    for i in 1:ndims(r)
        out[name(dims(r, i))] = vec(map(x -> x[i], itr))
    end
    return out
end

#-----------------------------------------------------------------------------# utils
function xyfeatures(r::Raster)
    x = repeat(dims(r, X), outer=length(r.dims[2]))
    y = repeat( r.dims[2].val.data, inner=length(r.dims[1]))
    (; x, y)
end

gaussian(u, k) = exp(-0.5 * (k * u) ^ 2)

# Return function for best-fit plane
function plane(r::Raster)
    z = vec(r)
    x, y = xyfeatures(r)
    β = hcat(x, y, x .* y) \ z
    (x, y) -> x * β[1] + y * β[2] + x * y * β[3]
end

#-----------------------------------------------------------------------------# AbstractGeoSurrogate
abstract type AbstractGeoSurrogate end

function Base.show(io::IO, o::T) where {T <: AbstractGeoSurrogate}
    print(io, T.name.name, " | ", GI.crs(o), " | ", GI.extent(o))
end

# Interface:
# GI.crs(::AbstractGeoSurrogate) = nothing
# GI.extent(::AbstractGeoSurrogate) = nothing


#-----------------------------------------------------------------------------# NoExtrapolation
struct NoExtrapolation end

#-----------------------------------------------------------------------------# RasterSurrogate
struct RasterSurrogate{F, C, E <: Extent} <: AbstractGeoSurrogate
    f::F
    crs::C
    extent::E
    function RasterSurrogate(r::Raster, alg = BSpline(Linear()), ext = NoExtrapolation())
        any(ismissing, r) && error("surrogates do not allow missing data")
        crs = GI.crs(r)
        extent = GI.extent(r)
        for (i, dim) in enumerate(r.dims)
            if step(dim.val.data) < 0
                r = reverse(r, dims=i)
            end
        end
        itp = interpolate(r.data, alg)
        itps = scale(itp, (x.val.data for x in r.dims)...)
        f = ext == NoExtrapolation() ? itps : extrapolate(itps, ext)
        return new{typeof(f), typeof(crs), typeof(extent)}(f, crs, extent)
    end
end
GI.crs(o::RasterSurrogate) = GI.crs(o.r)
GI.extent(o::RasterSurrogate) = o.extent
(o::RasterSurrogate)(x...) = o.f(x...)


#-----------------------------------------------------------------------------# GeometrySurrogate
struct GeometrySurrogate{F, T, C, E} <: AbstractGeoSurrogate
    f::F
    geomtrait::T
    crs::C
    extent::E
end
function GeometrySurrogate(geom, k)
    trt = GI.geomtrait(geom)
    f = get_fun(trt, geom, k)
    GeometrySurrogate(f, trt, GI.crs(geom), GI.extent(geom))
end
GI.crs(o::GeometrySurrogate) = o.crs
GI.extent(o::GeometrySurrogate) = o.extent
(o::GeometrySurrogate)(x...) = o.f(x...)

# Fallback
get_fun(::Any, geom, k) = (x, y) -> gaussian(GO.distance((x,y), geom), k)



end

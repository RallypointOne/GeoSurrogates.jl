using GeoSurrogates
using RasterDataSources, DataFrames, Rasters, ArchGDAL, Zygote
using Test
using StatsAPI: predict, fit!

import GeoInterface as GI

using p7zip_jll: p7zip


#-----------------------------------------------------------------------------# Test Data
url = "https://github.com/user-attachments/files/24169167/jl_d1CIih.zip"

function extract(file::AbstractString, dir::AbstractString = mktempdir())
    run(`$(p7zip()) x $file -o$dir -y`)
    return only(readdir(dir, join=true))
end

dir = extract(Base.download(url))

files = readdir(dir, join=true)

tif_file = only(filter(f -> endswith(f, ".tif"), files))

r = Raster(tif_file)

elev = let
    i = findfirst(==("US_ELEV2020"), r.dims[3].val)
    r[Band = i]
end

veg = let
    i = findfirst(==("US_250FBFM13"), r.dims[3].val)
    r[Band = i]
end

#-----------------------------------------------------------------------------# normalize
@testset "normalize" begin
    df = dropmissing!(DataFrame(elev))
    df2 = GeoSurrogates.normalize!(df)
    for col in names(df2)
        @test GeoSurrogates.is_normalized(df2[!, col])
    end

    # Test normalize(::Raster)
    r_norm = GeoSurrogates.normalize(elev)
    @test GeoSurrogates.is_normalized(r_norm.data)
    for dim in r_norm.dims
        if !DimensionalData.Lookups.iscategorical(dim)
            @test GeoSurrogates.is_normalized(dim.val.data)
        end
    end
end

#-----------------------------------------------------------------------------# RasterWrap
@testset "RasterWrap" begin
    rw = GeoSurrogates.RasterWrap(elev)
    @test GI.crs(rw) == GI.crs(elev)
    @test GI.extent(rw) == GI.extent(elev)
    for (coord, data) in zip(DimPoints(elev), elev)
        @test predict(rw, coord) == data
    end
end

#-----------------------------------------------------------------------------# CategoricalRasterWrap
@testset "CategoricalRasterWrap" begin
    crw = GeoSurrogates.CategoricalRasterWrap(veg)
    @test GI.crs(crw) == GI.crs(veg)
    @test GI.extent(crw) == GI.extent(veg)

    # Test that all unique levels are represented
    unique_levels = unique(skipmissing(veg.data))
    @test length(crw.geomwraps) == length(unique_levels)

    # Test that each class key is present in the geomwraps
    for k in keys(crw.geomwraps)
        @test k in unique_levels
    end
end

#-----------------------------------------------------------------------------# GeomWrap
@testset "GeomWrap" begin
    gw = GeoSurrogates.GeomWrap(GI.Point(0.0, 0.0))
    @test predict(gw, (0, 0)) == 1.0
    @test predict(gw, (1, 0)) ≈ GeoSurrogates.gaussian(1.0, 4)
end

#-----------------------------------------------------------------------------# Small synthetic raster for IDW/RBF/TPS tests
small_xs = range(-1, 1, length=15)
small_ys = range(-1, 1, length=15)
small_data = [exp(-(x^2 + y^2)) for x in small_xs, y in small_ys]
small_raster = Raster(small_data, (X(small_xs), Y(small_ys)))

#-----------------------------------------------------------------------------# IDW
@testset "IDW" begin
    m = IDW(small_raster)
    @test m isa GeoSurrogates.GeoSurrogate
    @test sprint(show, m) == "IDW"

    # Exact interpolation at known data points
    for (coord, val) in zip(DimPoints(small_raster), small_raster)
        @test predict(m, coord) ≈ val atol=1e-6
    end

    # Raster predict returns correct shape
    pred = predict(m, small_raster)
    @test size(pred) == size(small_raster)

    # fit! is a no-op
    @test fit!(m, small_raster) === m
end

#-----------------------------------------------------------------------------# RBF
@testset "RBF" begin
    m = RBF(small_raster; kernel=:gaussian, epsilon=1.0)
    @test m isa GeoSurrogates.GeoSurrogate
    @test sprint(show, m) == "RBF"

    # Exact interpolation at known data points
    for (coord, val) in zip(DimPoints(small_raster), small_raster)
        @test predict(m, coord) ≈ val atol=1e-6
    end

    # Raster predict returns correct shape
    pred = predict(m, small_raster)
    @test size(pred) == size(small_raster)

    # fit! is a no-op
    @test fit!(m, small_raster) === m

    # Test with poly_degree=1
    m2 = RBF(small_raster; kernel=:gaussian, epsilon=1.0, poly_degree=1)
    for (coord, val) in zip(DimPoints(small_raster), small_raster)
        @test predict(m2, coord) ≈ val atol=1e-6
    end
end

#-----------------------------------------------------------------------------# TPS
@testset "TPS" begin
    m = TPS(small_raster)
    @test m isa GeoSurrogates.GeoSurrogate
    @test sprint(show, m) == "TPS"

    # Exact interpolation at known data points (regularization=0)
    for (coord, val) in zip(DimPoints(small_raster), small_raster)
        @test predict(m, coord) ≈ val atol=1e-6
    end

    # Raster predict returns correct shape
    pred = predict(m, small_raster)
    @test size(pred) == size(small_raster)

    # fit! is a no-op
    @test fit!(m, small_raster) === m

    # Smoothing mode (regularization > 0) should still produce reasonable output
    m2 = TPS(small_raster; regularization=0.01)
    @test predict(m2, (0.0, 0.0)) isa Number
end

#-----------------------------------------------------------------------------# ImplicitTerrain
@testset "ImplicitTerrain" begin
    m = ImplicitTerrain.Model()

    predict(m, (0, 0))

    predict(m, elev)
end



# t = GeoSurrogates.ImplicitTerrainTrainer(raster = elev)

# @showprogress for _ in 1:10
#     GeoSurrogates.train_surface_model!(t; steps=3)
# end

# R = t.pyramid[1]
# x, y = GeoSurrogates.xyfeatures(R)

# XY = Float32.(vcat(x', y'))

# ẑ, st = Lux.apply(t.chain_s.chain, XY, t.chain_s.parameters, t.chain_s.states)

# Ẑ = reshape(ẑ, size(R))

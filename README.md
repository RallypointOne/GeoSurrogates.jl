# GeoSurrogates

[![Build Status](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/docs.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rallypointone.github.io/GeoSurrogates.jl/)

## Usage

```julia
using GeoSurrogates, Rasters

r = Raster(...)

rw = GeoSurrogates.RasterWrap(r)

predict(rw, x, y)  # Linear Interpolation
```

[![CI](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/CI.yml)
[![Docs Build](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/Docs.yml/badge.svg)](https://github.com/joshday/GeoSurrogates.jl/actions/workflows/Docs.yml)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue)](https://joshday.github.io/GeoSurrogates.jl/stable/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue)](https://joshday.github.io/GeoSurrogates.jl/dev/)

# GeoSurrogates.jl

## Usage

```julia
using GeoSurrogates, Rasters

r = Raster(...)

rw = GeoSurrogates.RasterWrap(r)

predict(rw, x, y)  # Linear Interpolation
```

[![CI](https://github.com/RallypointOne/GeoSurrogates.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/RallypointOne/GeoSurrogates.jl/actions/workflows/CI.yml)
[![Docs Build](https://github.com/RallypointOne/GeoSurrogates.jl/actions/workflows/Docs.yml/badge.svg)](https://github.com/RallypointOne/GeoSurrogates.jl/actions/workflows/Docs.yml)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue)](https://RallypointOne.github.io/GeoSurrogates.jl/stable/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue)](https://RallypointOne.github.io/GeoSurrogates.jl/dev/)

# GeoSurrogates.jl

**Implicit Neural Representations for Geospatial Data**

GeoSurrogates.jl creates surrogate models of geospatial data. It provides multiple approaches to learn continuous functions from spatial raster data, using both classical methods and neural networks to create compact, efficient representations of geographic phenomena.

## Features

- **Classical Surrogates** -- Linear regression, inverse distance weighting, radial basis functions, thin plate splines, and B-spline interpolation
- **Neural Network Surrogates** -- SIREN (Sinusoidal Representation Networks) for terrain, wind fields, and categorical data
- **Rasters.jl Integration** -- Works directly with `Raster` and `RasterStack` objects
- **Arbitrary Resolution** -- Predict at any resolution, not just the training grid
- **Memory Efficient** -- Neural networks as compact alternatives to storing full rasters
- **Composable** -- `AdditiveModel` for boosting-style ensembles of surrogates

## Installation

```julia
using Pkg
Pkg.add("GeoSurrogates")
```

## Quick Example

```julia
using GeoSurrogates, Rasters

# Load a raster
elev = Raster("path/to/elevation.tif")

# Create a simple interpolation-based surrogate
surrogate = RasterWrap(elev)

# Predict at any coordinate
predict(surrogate, (-105.5, 40.2))

# Or create a neural network surrogate for compression
model = ImplicitTerrain.Model()
fit!(model, normalize(elev); steps=1000)

# Predict on a new raster grid
predicted = predict(model, new_raster)
```

## Surrogate Types

| Type | Description | Use Case |
|------|-------------|----------|
| `RasterWrap` | B-spline interpolation | Fast exact interpolation |
| `LinReg` | Linear regression | Simple trend modeling |
| `IDW` | Inverse distance weighting | Scattered data interpolation |
| `RBF` | Radial basis functions | Flexible kernel interpolation |
| `TPS` | Thin plate splines | Smooth surface fitting |
| `GeomWrap` | Distance-based kernel | Geometry influence fields |
| `CategoricalRasterWrap` | Kernel smoothing | Categorical raster data |
| `ImplicitTerrain.Model` | Cascaded SIREN | Terrain compression |
| `WindSurrogate.WindSIREN` | SIREN for vectors | Wind field modeling |
| `CatSIREN.CatSIREN` | SIREN with softmax | Categorical classification |
| `AdditiveModel` | Boosted ensemble | Composing multiple surrogates |

## References

- Sitzmann et al. [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) (2020)
- ImplicitTerrain: [https://arxiv.org/abs/2406.00227](https://arxiv.org/abs/2406.00227)

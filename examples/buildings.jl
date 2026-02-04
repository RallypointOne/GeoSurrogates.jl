using GeoSurrogates, Rasters, GeoJSON, GLMakie, Extents
using Rasters: X, Y, dims

import GeoInterface as GI

#-----------------------------------------------------------------------------# Load buildings data
@info "Loading buildings data from Marshall fire area..."
buildings_path = expanduser("~/.julia/dev/MarshallWildfire/data/buildings.geojson")
fc = GeoJSON.read(buildings_path)

@info "Total buildings:" n=length(fc)

#-----------------------------------------------------------------------------# Select a subset of buildings for demonstration
n_buildings = 50
buildings = [GeoJSON.geometry(fc[i]) for i in 1:n_buildings]

# Compute combined extent of selected buildings
combined_ext = reduce(Extents.union, [GI.extent(b) for b in buildings])
ext = Extents.grow(combined_ext, 0.002)  # Add small buffer

@info "Using subset of buildings:" n=n_buildings extent=ext

#-----------------------------------------------------------------------------# Create MultiPolygon and single GeomWrap
@info "Creating MultiPolygon from buildings..."
multi_poly = GI.MultiPolygon(buildings)

@info "Creating single GeomWrap surrogate..."
kernel_width = 500  # Higher k = narrower falloff
surrogate = GeoSurrogates.GeomWrap(multi_poly; kernel=Base.Fix2(GeoSurrogates.gaussian, kernel_width))

#-----------------------------------------------------------------------------# Create reference raster for predictions
@info "Creating reference raster grid..."
xs = range(ext.X[1], ext.X[2], length=300)
ys = range(ext.Y[1], ext.Y[2], length=300)
ref_raster = Raster(zeros(length(xs), length(ys)), dims=(X(xs), Y(ys)))

#-----------------------------------------------------------------------------# Generate predictions
@info "Generating predictions..."
pred_raster = predict(surrogate, ref_raster)

#-----------------------------------------------------------------------------# Visualization
@info "Creating visualization..."

fig = Figure(size=(800, 700))

# Helper to plot filled polygon
function plot_polygon!(ax, poly; kwargs...)
    trait = GI.geomtrait(poly)
    if trait isa GI.PolygonTrait
        ring = GI.getexterior(poly)
        points = collect(GI.getpoint(ring))
        xs = [GI.x(p) for p in points]
        ys = [GI.y(p) for p in points]
        poly!(ax, Point2f.(zip(xs, ys)); kwargs...)
    end
end

# Main heatmap
ax = Axis(fig[1, 1], title="Building Proximity Influence (GeomWrap)",
          xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm = heatmap!(ax, pred_raster, colormap=:inferno)

# Overlay buildings with solid black fill
for building in buildings
    plot_polygon!(ax, building, color=:black)
end

Colorbar(fig[1, 2], hm, label="Combined Influence")

save(joinpath(@__DIR__, "buildings_geomwrap.png"), fig)
@info "Saved buildings_geomwrap.png"
display(fig)

@info "Done! Buildings GeomWrap example complete."

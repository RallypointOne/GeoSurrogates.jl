using GeoSurrogates, Rasters, GeoJSON, GLMakie, Extents
using Rasters: X, Y

import GeoInterface as GI

#-----------------------------------------------------------------------------# Load Marshall fire perimeter
@info "Loading Marshall fire perimeter geometry..."
marshall = GeoJSON.read(joinpath(@__DIR__, "marshall.geojson"))
poly = GI.getgeom(marshall, 1)

# Get extent and grow it for visualization
ext = Extents.grow(GI.extent(poly), 0.05)

@info "Geometry extent:" extent=ext

#-----------------------------------------------------------------------------# Create GeomWrap with different kernels
@info "Creating GeomWrap surrogates with different kernel bandwidths..."

# Default kernel (k=4)
gw_default = GeoSurrogates.GeomWrap(poly)

# Narrow kernel (k=20) - faster falloff
gw_narrow = GeoSurrogates.GeomWrap(poly; kernel=Base.Fix2(GeoSurrogates.gaussian, 20))

# Wide kernel (k=1) - slower falloff
gw_wide = GeoSurrogates.GeomWrap(poly; kernel=Base.Fix2(GeoSurrogates.gaussian, 1))

#-----------------------------------------------------------------------------# Create reference raster for predictions
@info "Creating reference raster grid..."
xs = range(ext.X[1], ext.X[2], length=200)
ys = range(ext.Y[1], ext.Y[2], length=200)
ref_raster = Raster(zeros(length(xs), length(ys)), dims=(X(xs), Y(ys)))

#-----------------------------------------------------------------------------# Generate predictions
@info "Generating predictions for each kernel..."
pred_default = predict(gw_default, ref_raster)
pred_narrow = predict(gw_narrow, ref_raster)
pred_wide = predict(gw_wide, ref_raster)

#-----------------------------------------------------------------------------# Visualization
@info "Creating visualization..."

fig = Figure(size=(1400, 900))

# Helper to plot polygon outline
function plot_polygon!(ax, poly; kwargs...)
    coords = GI.coordinates(poly)
    for ring in coords
        xs = [p[1] for p in ring]
        ys = [p[2] for p in ring]
        lines!(ax, xs, ys; kwargs...)
    end
end

# Row 1: Heatmaps with different kernels
ax1 = Axis(fig[1, 1], title="Wide kernel (k=1)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm1 = heatmap!(ax1, pred_wide, colormap=:viridis)
plot_polygon!(ax1, poly, color=:red, linewidth=2)
Colorbar(fig[1, 2], hm1, label="Influence")

ax2 = Axis(fig[1, 3], title="Default kernel (k=4)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm2 = heatmap!(ax2, pred_default, colormap=:viridis)
plot_polygon!(ax2, poly, color=:red, linewidth=2)
Colorbar(fig[1, 4], hm2, label="Influence")

ax3 = Axis(fig[1, 5], title="Narrow kernel (k=20)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())
hm3 = heatmap!(ax3, pred_narrow, colormap=:viridis)
plot_polygon!(ax3, poly, color=:red, linewidth=2)
Colorbar(fig[1, 6], hm3, label="Influence")

# Row 2: Cross-section comparison
mid_y_idx = size(ref_raster, 2) ÷ 2
x_coords = lookup(ref_raster, X)

ax_cross = Axis(fig[2, 1:3], title="Cross-section at Y midpoint",
                xlabel="Longitude", ylabel="Influence value")
lines!(ax_cross, x_coords, vec(pred_wide[:, mid_y_idx]), label="Wide (k=1)", linewidth=2)
lines!(ax_cross, x_coords, vec(pred_default[:, mid_y_idx]), label="Default (k=4)", linewidth=2)
lines!(ax_cross, x_coords, vec(pred_narrow[:, mid_y_idx]), label="Narrow (k=20)", linewidth=2)
axislegend(ax_cross, position=:rt)

# Row 2: Kernel function visualization
ax_kernel = Axis(fig[2, 4:6], title="Gaussian kernel: exp(-0.5 * (k * distance)²)",
                 xlabel="Distance from geometry", ylabel="Kernel value")
distances = range(0, 0.1, length=200)
lines!(ax_kernel, distances, [GeoSurrogates.gaussian(d, 1) for d in distances], label="k=1 (wide)", linewidth=2)
lines!(ax_kernel, distances, [GeoSurrogates.gaussian(d, 4) for d in distances], label="k=4 (default)", linewidth=2)
lines!(ax_kernel, distances, [GeoSurrogates.gaussian(d, 20) for d in distances], label="k=20 (narrow)", linewidth=2)
axislegend(ax_kernel, position=:rt)

save(joinpath(@__DIR__, "geomwrap_comparison.png"), fig)
@info "Saved geomwrap_comparison.png"
display(fig)

#-----------------------------------------------------------------------------# Contour plot
@info "Creating contour visualization..."

fig_contour = Figure(size=(800, 600))
ax_contour = Axis(fig_contour[1, 1], title="GeomWrap Influence Contours (default kernel)",
                  xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())

# Background heatmap
hm_bg = heatmap!(ax_contour, pred_default, colormap=:viridis)

# Contour lines at specific influence levels
contour!(ax_contour, lookup(ref_raster, X), lookup(ref_raster, Y), pred_default.data',
         levels=[0.1, 0.25, 0.5, 0.75, 0.9], color=:white, linewidth=1.5)

# Polygon outline
plot_polygon!(ax_contour, poly, color=:red, linewidth=3)

Colorbar(fig_contour[1, 2], hm_bg, label="Influence")

save(joinpath(@__DIR__, "geomwrap_contour.png"), fig_contour)
@info "Saved geomwrap_contour.png"
display(fig_contour)

@info "Done! GeomWrap example complete."

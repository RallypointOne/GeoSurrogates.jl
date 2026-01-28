using GeoSurrogates, Rasters, ArchGDAL, GLMakie, Statistics, Zygote, DataFrames
using p7zip_jll: p7zip
using Rasters: Band

#-----------------------------------------------------------------------------# Dataset
@info "Downloading terrain dataset..."
r = let
	url = "https://github.com/user-attachments/files/24169167/jl_d1CIih.zip"

	function extract(file::AbstractString, dir::AbstractString = mktempdir())
	    run(`$(p7zip()) x $file -o$dir -y`)
	    return only(readdir(dir, join=true))
	end

	dir = extract(Base.download(url))

	files = readdir(dir, join=true)

	tif_file = only(filter(f -> endswith(f, ".tif"), files))

	r = Raster(tif_file)

	i = findfirst(==("US_ELEV2020"), r.dims[3].val)
	r[Band = i][1:352, end-351:end]  # Crop to manageable size for example
end

@info "Loaded terrain raster" size=size(r) extrema=extrema(skipmissing(r))

#-----------------------------------------------------------------------------# Plot original terrain
@info "Plotting original terrain..."
fig_init = Figure(size=(800, 600))

ax_init = Axis(fig_init[1, 1], title="Original Terrain (Elevation)", xlabel="X", ylabel="Y", aspect=DataAspect())
hm_init = heatmap!(ax_init, r, colormap=:terrain)
Colorbar(fig_init[1, 2], hm_init, label="Elevation")

save(joinpath(@__DIR__, "terrain_original.png"), fig_init)
display(fig_init)

#-----------------------------------------------------------------------------# Normalize and prepare data
@info "Normalizing terrain data..."
r_norm = GeoSurrogates.normalize(r)

@info "Normalized data range:" range=extrema(skipmissing(r_norm))

#-----------------------------------------------------------------------------# Create and train ImplicitTerrain model
@info "Creating ImplicitTerrain.Model (surface + geometry MLPs)..."
model = GeoSurrogates.ImplicitTerrain.Model()

# Note: Paper uses 3000 steps per pyramid level; reduced here for speed
n_steps = 3000
@info "Training ImplicitTerrain model ($n_steps steps per pyramid level)..."
@time fit!(model, r_norm; steps=n_steps)

@info "Training complete!"

#-----------------------------------------------------------------------------# Predictions
@info "Generating predictions..."
z_pred = predict(model, r_norm)

# Calculate errors
z_error = r_norm .- z_pred
mae = mean(abs.(skipmissing(z_error)))
rmse = sqrt(mean(skipmissing(z_error).^2))

@info "Prediction errors:" mae=mae rmse=rmse

#-----------------------------------------------------------------------------# Visualization
@info "Creating comparison visualization..."

fig = Figure(size=(1400, 500))

# Original terrain
ax1 = Axis(fig[1, 1], title="Original (normalized)", xlabel="X", ylabel="Y", aspect=DataAspect())
hm1 = heatmap!(ax1, r_norm, colormap=:terrain)
Colorbar(fig[1, 2], hm1)

# Predicted terrain
ax2 = Axis(fig[1, 3], title="Predicted", xlabel="X", ylabel="Y", aspect=DataAspect())
hm2 = heatmap!(ax2, z_pred, colormap=:terrain)
Colorbar(fig[1, 4], hm2)

# Error map
ax3 = Axis(fig[1, 5], title="Error (Original - Predicted)", xlabel="X", ylabel="Y", aspect=DataAspect())
hm3 = heatmap!(ax3, z_error, colormap=:RdBu)
Colorbar(fig[1, 6], hm3)

save(joinpath(@__DIR__, "terrain_comparison.png"), fig)
@info "Displaying figure..."
display(fig)

#-----------------------------------------------------------------------------# Cross-section comparison
@info "Creating cross-section comparison..."

fig_cross = Figure(size=(1000, 400))

# Get middle row for cross-section
mid_y = size(r_norm, 2) ÷ 2
x_coords = lookup(r_norm, X)
z_orig_slice = vec(r_norm[:, mid_y])
z_pred_slice = vec(z_pred[:, mid_y])

ax_cross = Axis(fig_cross[1, 1], title="Cross-section at Y midpoint", xlabel="X coordinate", ylabel="Elevation (normalized)")
lines!(ax_cross, x_coords, z_orig_slice, label="Original", linewidth=2)
lines!(ax_cross, x_coords, z_pred_slice, label="Predicted", linewidth=2, linestyle=:dash)
axislegend(ax_cross, position=:rt)

save(joinpath(@__DIR__, "terrain_cross_section.png"), fig_cross)
display(fig_cross)

@info "Done! ImplicitTerrain example complete."

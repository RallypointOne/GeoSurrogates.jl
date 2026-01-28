using GeoSurrogates, Rasters, ArchGDAL, GLMakie, Statistics
using p7zip_jll: p7zip
using Rasters: Band

#-----------------------------------------------------------------------------# Dataset
@info "Downloading vegetation dataset..."
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

    i = findfirst(==("US_250FBFM13"), r.dims[3].val)
    r[Band = i][1:352, end-351:end]  # Crop to manageable size for example
end

categories = sort(unique(skipmissing(r)))
@info "Loaded vegetation raster" size=size(r) categories=categories

#-----------------------------------------------------------------------------# Plot original categorical raster
@info "Plotting original vegetation categories..."
fig_orig = Figure(size=(800, 600))

ax_orig = Axis(fig_orig[1, 1], title="Original Vegetation Categories (FBFM13)", xlabel="X", ylabel="Y", aspect=DataAspect())
hm_orig = heatmap!(ax_orig, r, colormap=:tab10)
Colorbar(fig_orig[1, 2], hm_orig, label="Category")

save(joinpath(@__DIR__, "vegetation_original.png"), fig_orig)
display(fig_orig)

#-----------------------------------------------------------------------------# Create CategoricalRasterWrap
@info "Creating CategoricalRasterWrap with kernel smoothing..."
crw = GeoSurrogates.CategoricalRasterWrap(r)

@info "CategoricalRasterWrap created" n_categories=length(crw.dict)

#-----------------------------------------------------------------------------# Predictions
@info "Generating predictions (this may take a moment)..."

# Predict on a coarser grid for speed
r_coarse = resample(r, size=size(r) .÷ 4)
@time pred_raster = predict(crw, r_coarse)

@info "Predictions complete" size=size(pred_raster)

#-----------------------------------------------------------------------------# Extract dominant category and entropy
@info "Computing dominant categories and entropy..."

# Get dominant category at each point
dominant = map(pred_raster.data) do d
    isempty(d) && return missing
    argmax(d)
end
dominant_raster = Raster(dominant, dims=Rasters.dims(pred_raster))

# Compute entropy (uncertainty) at each point
function entropy(d::Dict)
    isempty(d) && return missing
    probs = collect(values(d))
    -sum(p -> p > 0 ? p * log2(p) : 0.0, probs)
end

entropy_data = map(entropy, pred_raster.data)
entropy_raster = Raster(entropy_data, dims=Rasters.dims(pred_raster))

#-----------------------------------------------------------------------------# Visualization
@info "Creating visualization..."

fig = Figure(size=(1400, 500))

# Original categories (coarsened for comparison)
ax1 = Axis(fig[1, 1], title="Original Categories (coarsened)", xlabel="X", ylabel="Y", aspect=DataAspect())
hm1 = heatmap!(ax1, r_coarse, colormap=:tab10)
Colorbar(fig[1, 2], hm1, label="Category")

# Predicted dominant category
ax2 = Axis(fig[1, 3], title="Predicted Dominant Category", xlabel="X", ylabel="Y", aspect=DataAspect())
hm2 = heatmap!(ax2, dominant_raster, colormap=:tab10)
Colorbar(fig[1, 4], hm2, label="Category")

# Entropy (uncertainty)
ax3 = Axis(fig[1, 5], title="Prediction Entropy (Uncertainty)", xlabel="X", ylabel="Y", aspect=DataAspect())
hm3 = heatmap!(ax3, entropy_raster, colormap=:viridis)
Colorbar(fig[1, 6], hm3, label="Entropy (bits)")

save(joinpath(@__DIR__, "vegetation_comparison.png"), fig)
display(fig)

#-----------------------------------------------------------------------------# Probability maps for each category
@info "Creating probability maps for each category..."

fig_probs = Figure(size=(1200, 800))

for (i, cat) in enumerate(categories[1:min(6, length(categories))])
    row = (i - 1) ÷ 3 + 1
    col = (i - 1) % 3 + 1

    prob_data = map(d -> get(d, cat, 0.0), pred_raster.data)
    prob_raster = Raster(prob_data, dims=Rasters.dims(pred_raster))

    ax = Axis(fig_probs[row, col], title="P(Category = $cat)", xlabel="X", ylabel="Y", aspect=DataAspect())
    hm = heatmap!(ax, prob_raster, colormap=:blues, colorrange=(0, 1))
end

Colorbar(fig_probs[:, 4], limits=(0, 1), colormap=:blues, label="Probability")

save(joinpath(@__DIR__, "vegetation_probability_maps.png"), fig_probs)
display(fig_probs)

@info "Done! CategoricalRasterWrap example complete."

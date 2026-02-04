using GeoSurrogates, Rasters, ArchGDAL, GeoJSON, Dates, Extents, GLMakie, Statistics, Zygote, DataFrames, Tyler
using Dates: DateTime, Hour
using Rasters: Band

Rasters.checkmem!(false)

import RapidRefreshData as RR
import GeoInterface as GI
import GeoFormatTypes as GFT

#-----------------------------------------------------------------------------# Marshall fire
marshall = GeoJSON.read(joinpath(@__DIR__, "marshall.geojson"))
marshall_dates = (Date(2021, 12, 30), Date(2022, 1, 1))
marshall_ext = Extents.grow(GI.extent(marshall), 5.0)

#-----------------------------------------------------------------------------# HRRR wind data (3 days)
dset = RR.HRRRDataset(date=marshall_dates[1], cycle="00")

all_bands = RR.bands(dset)
u_band = only(filter(b -> b.variable == "UGRD" && b.level == "10 m above ground", all_bands))
v_band = only(filter(b -> b.variable == "VGRD" && b.level == "10 m above ground", all_bands))

function get_datetime(dset)
    DateTime(dset.date) + Hour(parse(Int, dset.cycle))
end

function get_uv_rasters(dset, u_band, v_band, extent)
    u = Raster(get(dset, [u_band]); checkmem=false)
    v = Raster(get(dset, [v_band]); checkmem=false)
    u = crop(resample(u, crs=GFT.EPSG("EPSG:4326")), to=extent)
    v = crop(resample(v, crs=GFT.EPSG("EPSG:4326")), to=extent)
    # Drop any remaining Band dimension to ensure consistent 2D rasters
    hasdim(u, Band) && (u = u[Band=1])
    hasdim(v, Band) && (v = v[Band=1])
    return u, v
end

# Collect rasters and timestamps
timestamps = DateTime[get_datetime(dset)]
u_rasters = Raster[]
v_rasters = Raster[]

u_first, v_first = get_uv_rasters(dset, u_band, v_band, marshall_ext)
push!(u_rasters, u_first)
push!(v_rasters, v_first)

while dset.date <= marshall_dates[2]
    global dset = RR.nextcycle(dset)
    @info "Retrieving HRRR data for $(dset.date) $(dset.cycle)..."
    push!(timestamps, get_datetime(dset))
    u_t, v_t = get_uv_rasters(dset, u_band, v_band, marshall_ext)
    push!(u_rasters, u_t)
    push!(v_rasters, v_t)
end

# Create RasterStack with Ti dimension by stacking along new Ti dimension
u_data = cat([r.data for r in u_rasters]..., dims=3)
v_data = cat([r.data for r in v_rasters]..., dims=3)

# Use dimensions from first raster, add Ti dimension
u_series = Raster(u_data, (dims(u_rasters[1])..., Ti(timestamps)))
v_series = Raster(v_data, (dims(v_rasters[1])..., Ti(timestamps)))
stack = RasterStack((UGRD=u_series, VGRD=v_series))

@info "Created stack with $(length(timestamps)) timesteps"

#-----------------------------------------------------------------------------# Fit WindSIREN
@info "Preparing wind data for training..."

# Get u and v components from the first timestep
u_raw = stack[:UGRD][Ti=1]
v_raw = stack[:VGRD][Ti=1]

# Plot initial wind field as arrows
@info "Plotting initial wind field..."
fig_init = Figure(size=(800, 600))

# Calculate wind speed for coloring
wind_speed = sqrt.(u_raw.^2 .+ v_raw.^2)

ax_init = Axis(fig_init[1, 1], title="Wind Field (10m above ground)", xlabel="Longitude", ylabel="Latitude", aspect=DataAspect())

# Show wind speed as background heatmap
hm = heatmap!(ax_init, wind_speed, colormap=:viridis)
Colorbar(fig_init[1, 2], hm, label="Wind Speed (m/s)")

# Subsample for arrow plot
df_init = DataFrame(u_raw)
xs_init = unique(df_init.X)
ys_init = unique(df_init.Y)
step_x_init = max(1, length(xs_init) ÷ 15)
step_y_init = max(1, length(ys_init) ÷ 15)
xs_sub_init = xs_init[1:step_x_init:end]
ys_sub_init = ys_init[1:step_y_init:end]

pts_init = [(x, y) for x in xs_sub_init, y in ys_sub_init]
coords_x_init = [p[1] for p in pts_init]
coords_y_init = [p[2] for p in pts_init]
u_init_vec = [u_raw[X=Near(x), Y=Near(y)] for (x, y) in pts_init]
v_init_vec = [v_raw[X=Near(x), Y=Near(y)] for (x, y) in pts_init]

arrows2d!(ax_init, vec(coords_x_init), vec(coords_y_init), vec(u_init_vec), vec(v_init_vec),
          lengthscale=0.003, color=:white)

save(joinpath(@__DIR__, "wind_original.png"), fig_init)
display(fig_init)

@info "Raw data ranges:" u_range=extrema(skipmissing(u_raw)) v_range=extrema(skipmissing(v_raw))

# Normalize rasters to [-1, 1] for SIREN training
u_norm = GeoSurrogates.normalize(u_raw)
v_norm = GeoSurrogates.normalize(v_raw)

@info "Normalized data ranges:" u_range=extrema(skipmissing(u_norm)) v_range=extrema(skipmissing(v_norm))

# Create and train WindSIREN model
@info "Creating WindSIREN model..."
model = GeoSurrogates.WindSurrogate.WindSIREN(
    hidden = 256,
    n_hidden = 3,
    ω0 = 10f0,
)

n_steps = 4000
eval_interval = 50
@info "Training WindSIREN for $n_steps steps (evaluating every $eval_interval steps)..."

# Train in chunks and compute MAE at intervals
losses = Float32[]
u_maes = Float32[]
v_maes = Float32[]
eval_steps = Int[]

@time for step in 1:eval_interval:n_steps
    chunk_size = min(eval_interval, n_steps - step + 1)
    chunk_losses = fit!(model, u_norm, v_norm; steps=chunk_size)
    append!(losses, chunk_losses)

    # Compute MAE on training data
    u_pred_eval, v_pred_eval = predict(model, u_norm)
    push!(u_maes, mean(abs.(skipmissing(u_norm .- u_pred_eval))))
    push!(v_maes, mean(abs.(skipmissing(v_norm .- v_pred_eval))))
    push!(eval_steps, step + chunk_size - 1)
end

@info "Training complete!" final_loss=losses[end] u_mae=u_maes[end] v_mae=v_maes[end]

# Plot loss and MAE over time
@info "Plotting training metrics..."
fig_loss = Figure(size=(800, 400))
ax_loss = Axis(fig_loss[1, 1], title="Training Metrics", xlabel="Step", ylabel="Value", yscale=log10)
lines!(ax_loss, 1:length(losses), losses, color=:blue, label="MSE Loss")
lines!(ax_loss, eval_steps, u_maes, color=:red, label="U MAE")
lines!(ax_loss, eval_steps, v_maes, color=:orange, label="V MAE")
axislegend(ax_loss, position=:rt)
save(joinpath(@__DIR__, "wind_loss.png"), fig_loss)
display(fig_loss)

#-----------------------------------------------------------------------------# Predictions
@info "Generating predictions..."

# Create higher resolution raster for predictions (4x resolution)
orig_size = size(u_norm)
hires_size = orig_size .* 4
u_norm_hires = resample(u_norm, size=hires_size)
u_pred, v_pred = predict(model, u_norm_hires)

# Calculate errors (resample original to match prediction resolution)
u_norm_compare = resample(u_norm, size=hires_size)
v_norm_compare = resample(v_norm, size=hires_size)
u_error = u_norm_compare .- u_pred
v_error = v_norm_compare .- v_pred

@info "Prediction errors:" u_mae=mean(abs.(skipmissing(u_error))) v_mae=mean(abs.(skipmissing(v_error)))

#-----------------------------------------------------------------------------# Visualization
@info "Creating visualization..."

fig = Figure(size=(1200, 800))

# Original u component
ax1 = Axis(fig[1, 1], title="U (original)", aspect=DataAspect())
hm1 = heatmap!(ax1, u_norm)
Colorbar(fig[1, 2], hm1)

# Predicted u component
ax2 = Axis(fig[1, 3], title="U (predicted)", aspect=DataAspect())
hm2 = heatmap!(ax2, u_pred)
Colorbar(fig[1, 4], hm2)

# Original v component
ax3 = Axis(fig[2, 1], title="V (original)", aspect=DataAspect())
hm3 = heatmap!(ax3, v_norm)
Colorbar(fig[2, 2], hm3)

# Predicted v component
ax4 = Axis(fig[2, 3], title="V (predicted)", aspect=DataAspect())
hm4 = heatmap!(ax4, v_pred)
Colorbar(fig[2, 4], hm4)

# Wind vector field comparison (subsampled for visibility)
ax5 = Axis(fig[3, 1:2], title="Wind vectors (original)", aspect=DataAspect())
ax6 = Axis(fig[3, 3:4], title="Wind vectors (predicted)", aspect=DataAspect())

# Get coordinates for quiver plot
df = DataFrame(u_norm)
xs = unique(df.X)
ys = unique(df.Y)

# Subsample for cleaner quiver plot
step_x = max(1, length(xs) ÷ 20)
step_y = max(1, length(ys) ÷ 20)

xs_sub = xs[1:step_x:end]
ys_sub = ys[1:step_y:end]

# Create meshgrid points
pts = [(x, y) for x in xs_sub, y in ys_sub]
coords_x = [p[1] for p in pts]
coords_y = [p[2] for p in pts]

# Get wind vectors at subsampled points
u_orig_vec = [u_norm[X=Near(x), Y=Near(y)] for (x, y) in pts]
v_orig_vec = [v_norm[X=Near(x), Y=Near(y)] for (x, y) in pts]
u_pred_vec = [u_pred[X=Near(x), Y=Near(y)] for (x, y) in pts]
v_pred_vec = [v_pred[X=Near(x), Y=Near(y)] for (x, y) in pts]

# Plot quiver
arrows2d!(ax5, vec(coords_x), vec(coords_y), vec(u_orig_vec), vec(v_orig_vec),
          lengthscale=0.04)
arrows2d!(ax6, vec(coords_x), vec(coords_y), vec(u_pred_vec), vec(v_pred_vec),
          lengthscale=0.04)

save(joinpath(@__DIR__, "wind_comparison.png"), fig)
@info "Displaying figure..."
display(fig)

#-----------------------------------------------------------------------------# U component comparison
@info "Creating U component comparison plot..."
fig_u = Figure(size=(500, 700))

ax_u1 = Axis(fig_u[1, 1], title="U (original)", aspect=DataAspect())
hm_u1 = heatmap!(ax_u1, u_norm)
Colorbar(fig_u[1, 2], hm_u1)

ax_u2 = Axis(fig_u[2, 1], title="U (predicted)", aspect=DataAspect())
hm_u2 = heatmap!(ax_u2, u_pred)
Colorbar(fig_u[2, 2], hm_u2)

save(joinpath(@__DIR__, "wind_u_comparison.png"), fig_u)
display(fig_u)

#-----------------------------------------------------------------------------# Wind arrows on map
@info "Creating wind arrows on map..."

# Get extent for map bounds
ext = GI.extent(u_raw)
west, east = ext.X
south, north = ext.Y

# Create Tyler map (it creates its own figure)
map_extent = Extent(X=(west, east), Y=(south, north))
m = Tyler.Map(map_extent; size=(1000, 800), crs=Tyler.wgs84)

# Wait for tiles to load
wait(m)

# Overlay Marshall fire perimeter
poly_coords = GI.coordinates(GI.getgeom(marshall, 1))
for ring in poly_coords
    xs = [p[1] for p in ring]
    ys = [p[2] for p in ring]
    lines!(m.axis, xs, ys, color=:red, linewidth=3)
end

# Overlay wind arrows (using same subsampled points)
arrows!(m.axis, vec(coords_x_init), vec(coords_y_init), vec(u_init_vec), vec(v_init_vec),
        lengthscale=0.005, color=:orange, linewidth=2, arrowsize=12)

save(joinpath(@__DIR__, "wind_map.png"), m.figure)
display(m.figure)

@info "Done! WindSIREN example complete."

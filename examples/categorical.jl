using GeoSurrogates, Landfire, OSMGeocoder, Rasters, ArchGDAL, GLMakie, DataFrames, Statistics, Zygote

import GeoInterface as GI

#-----------------------------------------------------------------------------# FBFM13 Fuel Model Names
const FBFM13_NAMES = Dict(
    1 => "Short Grass",
    2 => "Timber Grass/Understory",
    3 => "Tall Grass",
    4 => "Chaparral",
    5 => "Brush",
    6 => "Dormant Brush",
    7 => "Southern Rough",
    8 => "Closed Timber Litter",
    9 => "Hardwood Litter",
    10 => "Timber (Litter/Understory)",
    11 => "Light Slash",
    12 => "Medium Slash",
    13 => "Heavy Slash",
    91 => "Urban",
    92 => "Snow/Ice",
    93 => "Agriculture",
    98 => "Water",
    99 => "Barren",
)

#-----------------------------------------------------------------------------# Get Landfire FBFM13 data for Boulder County
@info "Fetching Landfire FBFM13 data for Boulder County, CO..."
boulder = geocode(county="Boulder", state="CO")

f13 = Landfire.products(layer="FBFM13", conus=true)
data = Landfire.Dataset(f13, boulder)
file = get(data)

r_full = Raster(file)

# Read into memory to avoid GDAL issues
r_full = read(r_full)

@info "Full raster size: $(size(r_full))"

# Subsample to ~100x100 for demo
step_x = max(1, size(r_full, 1) ÷ 100)
step_y = max(1, size(r_full, 2) ÷ 100)
r = r_full[1:step_x:end, 1:step_y:end]

@info "Raster size: $(size(r))"
@info "Unique fuel models: $(length(unique(skipmissing(r.data))))"

# Get unique classes and create color mapping
classes = sort(unique(skipmissing(r.data)))
n_classes = length(classes)
class_to_idx = Dict(c => i for (i, c) in enumerate(classes))

# Use a categorical colorscheme
colors = cgrad(:tab20, n_classes, categorical=true)

# Helper function to convert class values to indices for plotting
function class_to_index_raster(raster, class_to_idx)
    data = map(raster.data) do v
        ismissing(v) ? NaN32 : Float32(class_to_idx[v])
    end
    return Raster(data, dims=Rasters.dims(raster))
end

# Create legend elements
legend_elements = [PolyElement(color=colors[i]) for i in 1:n_classes]
legend_labels = [get(FBFM13_NAMES, Int(c), "FM $c") for c in classes]

#-----------------------------------------------------------------------------# Visualize original data
@info "Plotting original fuel model data..."
fig_orig = Figure(size=(1000, 600))
ax_orig = Axis(fig_orig[1, 1], title="FBFM13 Fuel Models (Original)", xlabel="X", ylabel="Y", aspect=DataAspect())
r_idx = class_to_index_raster(r, class_to_idx)
hm_orig = heatmap!(ax_orig, r_idx, colormap=colors, colorrange=(1, n_classes))
Legend(fig_orig[1, 2], legend_elements, legend_labels, "Fuel Model", framevisible=false)
save(joinpath(@__DIR__, "categorical_original.png"), fig_orig)
display(fig_orig)

#-----------------------------------------------------------------------------# Create and train CatSIREN
@info "Creating CatSIREN model..."
model = GeoSurrogates.CatSIREN.CatSIREN(r;
    hidden = 128,
    n_hidden = 3,
    ω0 = 30f0,
)

@info "Number of classes: $(length(model.classes))"
@info "Classes: $(model.classes)"

n_steps = 2000
eval_interval = 50
@info "Training CatSIREN for $n_steps steps..."

losses = Float32[]
accuracies = Float32[]
eval_steps = Int[]

@time for step in 1:eval_interval:n_steps
    chunk_size = min(eval_interval, n_steps - step + 1)
    chunk_losses = fit!(model, r; steps=chunk_size)
    append!(losses, chunk_losses)

    # Compute accuracy on training data
    pred_classes = GeoSurrogates.CatSIREN.predict_class(model, r)
    correct = sum(skipmissing(pred_classes.data .== r.data))
    total = sum(.!ismissing.(r.data))
    push!(accuracies, correct / total)
    push!(eval_steps, step + chunk_size - 1)

    if step % 500 == 1
        @info "Step $(step + chunk_size - 1): loss=$(round(losses[end], digits=4)), accuracy=$(round(accuracies[end] * 100, digits=1))%"
    end
end

@info "Training complete!" final_loss=losses[end] final_accuracy="$(round(accuracies[end] * 100, digits=1))%"

#-----------------------------------------------------------------------------# Plot training metrics
@info "Plotting training metrics..."
fig_loss = Figure(size=(800, 400))

ax_loss = Axis(fig_loss[1, 1], title="Training Loss", xlabel="Step", ylabel="Cross-Entropy Loss")
lines!(ax_loss, 1:length(losses), losses, color=:blue)

ax_acc = Axis(fig_loss[1, 2], title="Training Accuracy", xlabel="Step", ylabel="Accuracy")
lines!(ax_acc, eval_steps, accuracies .* 100, color=:green)

save(joinpath(@__DIR__, "categorical_training.png"), fig_loss)
display(fig_loss)

#-----------------------------------------------------------------------------# Generate predictions
@info "Generating predictions..."

# Predict class labels
pred_classes = GeoSurrogates.CatSIREN.predict_class(model, r)

# Predict at higher resolution (use original raster at finer resolution)
step_hires = max(1, step_x ÷ 2)
r_hires = r_full[1:step_hires:end, 1:step_hires:end]
pred_hires = GeoSurrogates.CatSIREN.predict_class(model, r_hires)

#-----------------------------------------------------------------------------# Visualize comparison
@info "Creating comparison visualization..."
fig = Figure(size=(900, 700))

ax1 = Axis(fig[1, 1], title="Original Fuel Models", aspect=DataAspect())
r_idx = class_to_index_raster(r, class_to_idx)
hm1 = heatmap!(ax1, r_idx, colormap=colors, colorrange=(1, n_classes))

ax2 = Axis(fig[2, 1], title="Predicted Fuel Models", aspect=DataAspect())
pred_idx = class_to_index_raster(pred_classes, class_to_idx)
hm2 = heatmap!(ax2, pred_idx, colormap=colors, colorrange=(1, n_classes))

Legend(fig[1:2, 2], legend_elements, legend_labels, "Fuel Model", framevisible=false)

save(joinpath(@__DIR__, "categorical_comparison.png"), fig)
display(fig)

#-----------------------------------------------------------------------------# Show probability distribution for a sample point
@info "Showing probability distribution for sample points..."

# Get center coordinates
df = DataFrame(r)
center_x = median(df.X)
center_y = median(df.Y)

# Normalize coordinates for prediction
x_norm = GeoSurrogates.normalize(df.X)
y_norm = GeoSurrogates.normalize(df.Y)
center_x_norm = median(x_norm)
center_y_norm = median(y_norm)

probs = predict(model, (center_x_norm, center_y_norm))

# Plot probability distribution
fig_probs = Figure(size=(900, 400))
ax_probs = Axis(fig_probs[1, 1],
    title="Class Probabilities at Center Point",
    xlabel="Fuel Model",
    ylabel="Probability",
    xticks=(1:length(model.classes), legend_labels),
    xticklabelrotation=π/4
)
barplot!(ax_probs, 1:length(probs), probs, color=[colors[i] for i in 1:length(probs)])

save(joinpath(@__DIR__, "categorical_probabilities.png"), fig_probs)
display(fig_probs)

@info "Sum of probabilities: $(sum(probs))"  # Should be 1.0

#-----------------------------------------------------------------------------# Higher resolution prediction
@info "Creating high-resolution prediction..."
fig_hires = Figure(size=(1000, 600))
ax_hires = Axis(fig_hires[1, 1], title="High-Resolution Predicted Fuel Models (2x)", aspect=DataAspect())
pred_hires_idx = class_to_index_raster(pred_hires, class_to_idx)
hm_hires = heatmap!(ax_hires, pred_hires_idx, colormap=colors, colorrange=(1, n_classes))
Legend(fig_hires[1, 2], legend_elements, legend_labels, "Fuel Model", framevisible=false)
save(joinpath(@__DIR__, "categorical_hires.png"), fig_hires)
display(fig_hires)

@info "Done! CatSIREN example complete."

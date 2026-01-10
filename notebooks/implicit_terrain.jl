### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ c4c0a6bd-bb24-4bb5-ba6c-19eca49ccfa0
begin
	using Pkg 
	
	Pkg.activate("..")
	
	using GeoSurrogates, Rasters, ArchGDAL, GLMakie, PlutoUI, Zygote
		
	import GeoInterface as GI

	using StatsAPI: predict, fit!
	using p7zip_jll: p7zip

	PlutoUI.TableOfContents()
end

# ╔═╡ 0c2b386e-c03f-4502-b6a5-7e139d2c68d9
md"# Load Elevation Data"

# ╔═╡ f9e21a90-ebe5-11f0-b6d2-dbc6faa73369
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
	r[Band = i]
end

# ╔═╡ 1a71102b-6c8c-4b86-a78e-311d96385892
rnorm = reverse(GeoSurrogates.normalize(r), dims=1)

# ╔═╡ e9815bca-22d1-44cd-b767-d44fd8bc62f2
md"## Plot of Input Data"

# ╔═╡ d0ff643c-77d8-4ca8-9b5e-78e80fb840bf
plot(rnorm)

# ╔═╡ 999d088f-ddd3-4424-be78-ec3c7e6dd369
md"# Gaussian Pyramid"

# ╔═╡ 0f91a574-c623-4fb2-a3f5-224d419c7ab0
pyr = ImplicitTerrain._gaussian_pyramid(rnorm)

# ╔═╡ a46ecc6b-926d-40da-ae31-a4eadc2968a8
let 
	fig = Figure()
	ax = Axis(fig[1, 1], title="Original")
	ax2 = Axis(fig[1, 2], title="Smoothed/Downsampled 1")
	ax3 = Axis(fig[2, 1], title="Smoothed/Downsampled 2")
	ax4 = Axis(fig[2, 2], title="Smoothed/Downsampled 3")
	plot!(ax, pyr[1])
	plot!(ax2, pyr[2])
	plot!(ax3, pyr[3])
	plot!(ax4, pyr[4])
	hidedecorations!.([ax, ax2, ax3, ax4])

	fig
end

# ╔═╡ 42bd74c8-8ed2-430b-81e5-09b0d17149c7
md"# Create Model"

# ╔═╡ ae05e390-5294-4a3d-bf43-ace66455bbce
m = ImplicitTerrain.Model()

# ╔═╡ af0d5792-190d-4c24-a50b-1945b4d54f64
md"## Plot with default weights"

# ╔═╡ 5c22da31-f29f-4034-a75d-1e50a6fd71c7
plot(predict(m, rnorm))

# ╔═╡ 1ef9cdae-0cd9-4bfb-ba58-4e5b0fc31301
md"# Fit Model"

# ╔═╡ 4c155c6c-f58c-4bf8-a755-ece43b13b58b
function make_anim(model, raster, name)
	r = GeoSurrogates.normalize(raster)
	x = ImplicitTerrain.features(r)
	y = vec(r)

	plot_data = Observable(predict(model, r))
	title = Observable("")
	fig, ax, plt = plot(plot_data; axis=(; title))
	steps_per_frame = 100

	record(fig, name * ".gif", 1:30; framerate = 10) do i 
		fit!(model, x, y; steps=steps_per_frame)
		title[] = "$name Iteration: $(i * steps_per_frame)"
		plot_data[] = predict(model, pyr[1])
	end
	
end

# ╔═╡ ddb9908a-21d0-4d45-ad69-06fcab057c18
md"## Learning on Pyramid Level 4 (top)"

# ╔═╡ db0f2f67-6c86-4440-a00a-7f909bf33d00
LocalResource(make_anim(m.surface, pyr[4], "Surface - Level 4"))

# ╔═╡ 58906e6a-1218-469d-b60a-6099e3cc5f91
LocalResource(make_anim(m.surface, pyr[3], "Surface - Level 3"))

# ╔═╡ a2584656-ad34-4db8-833c-ec7346fc6fda
LocalResource(make_anim(m.surface, pyr[2], "Surface - Level 2"))

# ╔═╡ a0f19d55-a0a1-403c-929f-bd13f89be75d
# let
# 	resid = rnorm - predict(m.surface, rnorm)
# 	LocalResource(make_anim(m.geometry, resid, "Geometry - Level 1"))
# end

# ╔═╡ Cell order:
# ╟─c4c0a6bd-bb24-4bb5-ba6c-19eca49ccfa0
# ╟─0c2b386e-c03f-4502-b6a5-7e139d2c68d9
# ╟─f9e21a90-ebe5-11f0-b6d2-dbc6faa73369
# ╠═1a71102b-6c8c-4b86-a78e-311d96385892
# ╟─e9815bca-22d1-44cd-b767-d44fd8bc62f2
# ╠═d0ff643c-77d8-4ca8-9b5e-78e80fb840bf
# ╟─999d088f-ddd3-4424-be78-ec3c7e6dd369
# ╟─0f91a574-c623-4fb2-a3f5-224d419c7ab0
# ╟─a46ecc6b-926d-40da-ae31-a4eadc2968a8
# ╟─42bd74c8-8ed2-430b-81e5-09b0d17149c7
# ╠═ae05e390-5294-4a3d-bf43-ace66455bbce
# ╟─af0d5792-190d-4c24-a50b-1945b4d54f64
# ╠═5c22da31-f29f-4034-a75d-1e50a6fd71c7
# ╟─1ef9cdae-0cd9-4bfb-ba58-4e5b0fc31301
# ╟─4c155c6c-f58c-4bf8-a755-ece43b13b58b
# ╟─ddb9908a-21d0-4d45-ad69-06fcab057c18
# ╟─db0f2f67-6c86-4440-a00a-7f909bf33d00
# ╟─58906e6a-1218-469d-b60a-6099e3cc5f91
# ╟─a2584656-ad34-4db8-833c-ec7346fc6fda
# ╠═a0f19d55-a0a1-403c-929f-bd13f89be75d

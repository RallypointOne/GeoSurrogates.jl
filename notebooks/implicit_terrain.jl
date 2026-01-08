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
rnorm = GeoSurrogates.normalize(r)

# ╔═╡ e9815bca-22d1-44cd-b767-d44fd8bc62f2
md"## Plot of Input Data"

# ╔═╡ d0ff643c-77d8-4ca8-9b5e-78e80fb840bf
plot(rnorm)

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

# ╔═╡ b9e0ef1c-ac5e-4924-9077-8a09ea0a1441
fit!(m, r; steps=3000)

# ╔═╡ cb7120ce-bd4f-43e1-a49b-325191f12b22
plot(predict(m, rnorm))

# ╔═╡ Cell order:
# ╟─c4c0a6bd-bb24-4bb5-ba6c-19eca49ccfa0
# ╟─0c2b386e-c03f-4502-b6a5-7e139d2c68d9
# ╟─f9e21a90-ebe5-11f0-b6d2-dbc6faa73369
# ╠═1a71102b-6c8c-4b86-a78e-311d96385892
# ╟─e9815bca-22d1-44cd-b767-d44fd8bc62f2
# ╠═d0ff643c-77d8-4ca8-9b5e-78e80fb840bf
# ╟─42bd74c8-8ed2-430b-81e5-09b0d17149c7
# ╠═ae05e390-5294-4a3d-bf43-ace66455bbce
# ╟─af0d5792-190d-4c24-a50b-1945b4d54f64
# ╠═5c22da31-f29f-4034-a75d-1e50a6fd71c7
# ╟─1ef9cdae-0cd9-4bfb-ba58-4e5b0fc31301
# ╠═b9e0ef1c-ac5e-4924-9077-8a09ea0a1441
# ╠═cb7120ce-bd4f-43e1-a49b-325191f12b22

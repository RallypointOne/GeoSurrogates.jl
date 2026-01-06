### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f2146508-d20a-11f0-b7ed-6d3abc53e8a6
begin
	using Pkg 
	Pkg.activate("..")
	using GeoSurrogates, Rasters, RasterDataSources, ArchGDAL, GeoJSON, Landfire, 
		OSMGeocoder, PlutoUI, Scratch, GLMakie, GeoMakie, Extents

	Rasters.checkmem!(false)

	PlutoUI.TableOfContents()
end

# ╔═╡ 0574ee3c-3485-42f5-80ed-17c9dabf3854
md"# Data"

# ╔═╡ 60ff7421-5538-4fbc-bb58-7493f48a6349
path = "/Users/joshday/datasets/LF2023_FBFM13_240_CONUS/Tif/LC23_F13_240.tif"

# ╔═╡ 1c6c281a-6c9c-4177-be06-ddc8c1d8e88b
Base.format_bytes(filesize(path))

# ╔═╡ 81201a09-8760-40d4-b748-1fddd840a33e
r = Raster(path)

# ╔═╡ 693945fc-facb-4d39-87fc-08111bc71c2e
area = geocode(city="Boulder", state="CO")

# ╔═╡ 84e95b08-56ff-464a-a007-ec0a1050b896
ext = extent(area[1])

# ╔═╡ Cell order:
# ╟─f2146508-d20a-11f0-b7ed-6d3abc53e8a6
# ╟─0574ee3c-3485-42f5-80ed-17c9dabf3854
# ╠═60ff7421-5538-4fbc-bb58-7493f48a6349
# ╠═1c6c281a-6c9c-4177-be06-ddc8c1d8e88b
# ╠═81201a09-8760-40d4-b748-1fddd840a33e
# ╠═693945fc-facb-4d39-87fc-08111bc71c2e
# ╠═84e95b08-56ff-464a-a007-ec0a1050b896

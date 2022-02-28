# This code has been tested with Julia version 1.5.1
using DataFrames # Tested with version 1.2.2
using CSV # Tested with version 0.9.9
using Plots # Tested with version 1.21.3
using CategoricalArrays # Tested with version 0.10.0
using LaTeXStrings # Tested with version 1.2.1
using Statistics
using StatsBase # Tested with version 0.33.10
pyplot() # Tested with version 2.9.0

# Generate all figures and tables.
# The scripts sim_exhaustive.jl, sim_algorithm.jl and sim_crossestimation.R
# have to be run first for all relevant parameters
# to generate all data.

function plot_k(df, varlist, hetero, kmax, yl, sz, te, xt, yt)
    dft = df[(df[!, :hetero] .== hetero) .& (df[!, :tau] .== te), :]
    gdf = groupby(dft, [:corr, :k])
    df_agg = combine(gdf, valuecols(gdf) .=> mean, nrow)
    df_names = [replace(i, "_mean" => "") for i = names(df_agg)]
    rename!(df_agg, Symbol.(df_names))
    sort!(df_agg, :k)
    dfc = df_agg[df_agg[!, :corr] .== true, :]
    dfuc = df_agg[df_agg[!, :corr] .== false, :]
    a = 10
    b = 10
    p1 = plot(dfc[!, :k], [dfc[!, varlist[1]], dfc[!, varlist[2]], dfc[!, varlist[3]],
              dfc[!, varlist[4]]], legend=false, title="Correlated covariates",
              label=["Difference-in-means" "Regression" "PCA alg" "Cross-estimation"],
              xlabel="",xtickfontsize=a,
              ytickfontsize=a,xguidefontsize=a,legendfontsize=a,
              titlefontsize=b, grid=false, ylims=yl,
              xlims=(0, kmax * 1.05), xticks=xt, yticks=[])
    p2 = plot(dfuc[!, :k], [dfuc[!, varlist[1]], dfuc[!, varlist[2]], dfuc[!, varlist[3]],
              dfuc[!, varlist[4]]], legend=false, title="Uncorrelated covariates",
              label=["Difference-in-means" "Regression" "PCA alg" "Cross-estimation"],
              xlabel="",xtickfontsize=a,
              ytickfontsize=a,xguidefontsize=a,legendfontsize=a,
              titlefontsize=b, grid=false, ylims=yl,
              xlims=(0, kmax * 1.05), xticks=xt, yticks=yt)
    legend = plot([0 0 0 0], showaxis = false, grid = false,
          label = ["Difference-in-means" "Regression" "PCA alg" "Cross-estimation"],
          legendfontsize=b)
    p3 = plot(p2, p1, layout = @layout([A B]),size = sz)
    return p3
end

function plot_cond_m(df, varlist, klist, hetero, corr, yl, sz, lg, te, xt, yt)
    nk = length(klist)
    p = Array{Any}(undef, nk)
    dfn = df[(df[!, :corr] .== corr) .& (df[!, :hetero] .== hetero) .&
             (df[!, :tau] .== te), :]
    sort!(dfn, [:k, :quantile])
    a = 10
    b = 10
    for i = 1:nk
        if i != 1
            ytt = []
        else
            ytt = deepcopy(yt)
        end
        dfk = dfn[dfn[!, :k] .== klist[i], :]
        p[i] = plot(dfk[!, :quantile], [dfk[!, varlist[1]], dfk[!, varlist[2]],
                                     dfk[!, varlist[3]], dfk[!, varlist[4]]],
                  legend=false,
                  label=["Difference-in-means" "Regression" "PCA alg" "Cross-estimation"],
                  title=string("K=", klist[i]), xlabel="", ylims=yl,
                  xtickfontsize=a,ytickfontsize=a,xguidefontsize=a,
                  titlefontsize=b, grid=false, xlims=(0, 105),
                  xticks=xt, yticks=ytt)
    end
    if lg == true
        legend = plot([0 0 0 0], showaxis = false, grid = false,
                      label = ["Difference-in-means" "Regression" "PCA alg" "Cross-estimation"],
                      legendfontsize=b, legend=:topright)
        return plot(p[1], p[2], p[3], legend,
                    layout = @layout([A B C ; D{.35h}]), size = sz)
    elseif lg == false
        return plot(p[1], p[2], p[3],
                    layout = @layout([A B C]), size = sz)
    end
end

function get_quantile_row(var, df)
    s = string(" & ", string(round(mean(df[!, var]), digits=3)))
    for i = 1:5
        s = string(s, " & ", string(round(df[!, var][i], digits=3)))
    end
    return string(s, " \\\\\n")
end

function get_panel(di, df, k)
    dfk = df[df[!, :k] .== k, :]
    s = string("\\multicolumn{7}{l}{\\(K=", string(k), "\\)}\\\\\n")
    for (key, val) = di
        s = string(s, val)
        s = string(s, get_quantile_row(key, dfk))
    end
    s = string(s, "\\addlinespace\n\\addlinespace")
    return s
end

function get_table(di, df, klist, hetero, tau)
    dft = df[(df[!, :hetero] .== hetero) .& (df[!, :tau] .== tau), :]
    s = "\\begin{tabular*}{\\textwidth}{l @{\\extracolsep{\\fill}} cccccc}"
    s = string(s, "\n\\toprule\nQuintiles & All & 1st & 2nd & 3rd & 4th & 5th \\\\")
    s = string(s, "\n\\midrule")
    dfuc = dft[(dft[!, :corr] .== false), :]
    s = string(s, " & \\multicolumn{6}{c}{Uncorrelated covariates} \\\\\n")
    s = string(s, "\\cmidrule(lr){2-7}\n")
    for k = klist
        s = string(s, get_panel(di, dfuc, k))
    end
    dfc = dft[(dft[!, :corr] .== true), :]
    s = string(s, " & \\multicolumn{6}{c}{Correlated covariates} \\\\\n")
    s = string(s, "\\cmidrule(lr){2-7}\n")
    for k = klist
        s = string(s, get_panel(di, dfc, k))
    end
    return string(s, "\\bottomrule\n\\end{tabular*}")
end

function gen_simple_sim_plot(df_agg_np_complete, df_mean_np_complete)
    c1 = round(abs(df_mean_np_complete[!, :DM_est][1]), digits=2)
    c2 = round(abs(df_mean_np_complete[!, :OLS_est][1]), digits=2)
    c3 = round(df_mean_np_complete[!, :DM_var][1], digits=2)
    c4 = round(df_mean_np_complete[!, :OLS_var][1], digits=2)
    c5 = round(df_mean_np_complete[!, :s_DM][1], digits=2)
    c6 = round(df_mean_np_complete[!, :s_OLS][1], digits=2)
    c7 = round(df_mean_np_complete[!, :DM_mse][1], digits=2)
    c8 = round(df_mean_np_complete[!, :OLS_mse][1], digits=2)

    s1 = plot(df_agg_np_complete[!, :Delta],
              [df_agg_np_complete[!, :DM_est], df_agg_np_complete[!, :OLS_est]],
              legend=false,
              label=["Difference-in-means" "Regression"], title="Point estimate",
              xlabel="",xtickfontsize=10,ytickfontsize=10,xguidefontsize=10,
              titlefontsize=12, grid=false)
    s2 = plot(df_agg_np_complete[!, :Delta],
              [df_agg_np_complete[!, :DM_var], df_agg_np_complete[!, :OLS_var]],
              legend=false,
              label=["Difference-in-means" "Regression"], title="Variance",
              xlabel="",xtickfontsize=10,ytickfontsize=10,xguidefontsize=10,
              titlefontsize=12, grid=false, ylims=(0, .4))
    s3 = plot(df_agg_np_complete[!, :Delta],
              [df_agg_np_complete[!, :s_DM], df_agg_np_complete[!, :s_OLS]],
              legend=false,
              label=["Difference-in-means" "Regression"],
              title="Statistical significance",
              xlabel="",xtickfontsize=10,ytickfontsize=10,xguidefontsize=10,
              titlefontsize=12, grid=false, ylims=(0, .5))
    s4 = plot(df_agg_np_complete[!, :Delta],
              [df_agg_np_complete[!, :DM_mse], df_agg_np_complete[!, :OLS_mse]],
              legend=false,
              label=["Difference-in-means" "Regression"],
              title="Mean squared error",
              xlabel="",xtickfontsize=10,ytickfontsize=10,xguidefontsize=10,
              titlefontsize=12, grid=false, ylims=(0, 1.7))
    t = plot(showaxis = false, grid = false; annotations=[
        (0.138,0.85, text(latexstring("E(\\hat\\tau_{DM})=",c1),12)),
        (0.118,0.7, text(latexstring("E(\\hat\\tau_z)=",c2),12)),
        (0.158,0.55, text(latexstring("V(\\hat\\tau_{DM})=",c3),12)),
        (0.140,0.4, text(latexstring("V(\\hat\\tau_z)=",c4),12)),
        (0.71,0.85, text(latexstring("E(\\pi_{DM}<0.05)=",c5),12)),
        (0.69,0.7, text(latexstring("E(\\pi_z<0.05)=",c6),12)),
        (0.656,0.55, text(latexstring("MSE(\\hat\\tau_{DM})=",c7),12)),
        (0.638,0.4, text(latexstring("MSE(\\hat\\tau_z)=",c8),12)),
        ] )
    legend = plot([0 0], showaxis = false, grid = false,
                  label = ["Difference-in-means" "Regression"],
                  legendfontsize=12)
    return s1, s3, s2, s4, t, legend
end

df_agg_np_complete = CSV.read("np_complete_agg.csv", DataFrame)
df_mean_np_complete = CSV.read("np_complete_mean.csv", DataFrame)
df_agg_np_complete_single = CSV.read("np_complete_agg_single.csv", DataFrame)
df_mean_np_complete_single = CSV.read("np_complete_mean_single.csv", DataFrame)

s1, s3, s2, s4, t, legend = gen_simple_sim_plot(df_agg_np_complete_single,
                                                df_mean_np_complete_single)
plot(s1, s3, s2, s4, t, legend, layout = @layout([A B; C D ; E F]),
     size = (700, 600))
savefig(string("simple_sim_single.pdf"))

s1, s3, s2, s4, t, legend = gen_simple_sim_plot(df_agg_np_complete,
                                                df_mean_np_complete)
plot(s1, s3, s2, s4, t, legend, layout = @layout([A B; C D ; E F]),
     size = (700, 600))
savefig(string("simple_sim.pdf"))

df_np = CSV.read("sim_np.csv", DataFrame)
df_tibs = CSV.read("sim_tibs.csv", DataFrame)
rename!(df_tibs, Dict(:M => :M_tibs, :SATE => :SATE_tibs))
df_tibs = select(df_tibs, Not([:reps_in_sample, :reps, :n, :n0]))
df_np = outerjoin(df_np, df_tibs, on = [:quantile, :corr, :hetero, :k, :tau])
df_np = df_np[df_np[!, :k] .!= 1, :]

[df_np[!, var] = coalesce.(df_np[!, var], NaN) for var = propertynames(df_np)]

df_np2 = deepcopy(df_np)
df_np2[!, :cut] = cut(df_np2[!, :quantile], 5)
gdf = groupby(df_np2, [:cut, :k, :corr, :hetero, :tau])

df_agg = combine(gdf, [:s_DM, :s_OLS, :s_ALG, :s_TIBS, :k, :NR_P] .=> mean, nrow)

d = Dict(:s_DM_mean => "Difference-in-means",
         :s_OLS_mean => "Regression",
         :s_ALG_mean => "PCA alg",
         :s_TIBS_mean => "Cross-estimation")

a = get_table(d, df_agg, [10, 20, 30], false, 0.0)
io = open("size_homo.tex", "w")
write(io, a)
close(io)

a = get_table(d, df_agg, [10, 20, 30], false, 1.0)
io = open("power_homo.tex", "w")
write(io, a)
close(io)

a = get_table(d, df_agg, [5, 10, 15], true, 0.0)
io = open("size_hetero.tex", "w")
write(io, a)
close(io)

a = get_table(d, df_agg, [5, 10, 15], true, 1.0)
io = open("power_hetero.tex", "w")
write(io, a)
close(io)

plot_k(df_np, [:DM_mse, :OLS_mse, :ALG_mse, :TIBS_mse], false, 40, (0, .55),
       (467, 225), 0.0, [0, 20, 40], [0, 0.5])
savefig(string("mse_homo_k.pdf"))

plot_k(df_np, [:DM_mse, :OLS_mse, :ALG_mse, :TIBS_mse], true, 20, (0, .55),
       (467, 225), 0.0, [0, 20, 40], [0, 0.5])
savefig(string("mse_hetero_k.pdf"))

plot_cond_m(df_np, [:DM_mse, :OLS_mse, :ALG_mse, :TIBS_mse], [10, 20, 30],
            false, false, (0, 0.55), (700, 225), false, 0.0, [0, 50, 100],
            [0, 0.5])
savefig(string("mse_homo_perc_uncorr.pdf"))

plot_cond_m(df_np, [:DM_mse, :OLS_mse, :ALG_mse, :TIBS_mse], [10, 20, 30],
            false, true, (0, 0.55), (700, 350), true, 0.0, [0, 50, 100],
            [0, 0.5])
savefig(string("mse_homo_perc_corr.pdf"))

plot_cond_m(df_np, [:DM_mse, :OLS_mse, :ALG_mse, :TIBS_mse], [5, 10, 15],
            true, false, (0, 0.55), (700, 225), false, 0.0, [0, 50, 100],
            [0, 0.5])
savefig(string("mse_hetero_perc_uncorr.pdf"))

plot_cond_m(df_np, [:DM_mse, :OLS_mse, :ALG_mse, :TIBS_mse], [5, 10, 15],
            true, true, (0, 0.55), (700, 350), true, 0.0, [0, 50, 100],
            [0, 0.5])
savefig(string("mse_hetero_perc_corr.pdf"))

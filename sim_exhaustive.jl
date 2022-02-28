# This code has been tested with Julia version 1.5.1
using Combinatorics # Tested with version 1.0.2
using Distributions # Tested with version 0.23.8
using DataFrames # Tested with version 1.2.2
using CSV # Tested with version 0.9.9
using Random
using Dates
using GLM # Tested with version 1.5.1
using RandomCorrelationMatrices # Tested with version 1.0.0
using CategoricalArrays # Tested with version 0.10.0

function gen_𝒲()
    # Generate the set of all possible treatment assignment vectors
    𝒲 = Array[]
    for i = combinations(1:n, n₁)
        W = zeros(n)
        W[i] .+= 1
        W = reshape(W, n, 1)
        push!(𝒲, Int.(W))
    end
    return 𝒲
end

function sim_reg()
    # Function that goes through all assignment vectors and calculate
    # the difference-in-means and OLS estimates of the sample average
    # treatment effect. P-values for the Neyman-Pearson tests and
    # Fisher's exact test are also returned.
    beta = sqrt((rsq/100)/(1-(rsq/100)))
    Z = rand(Normal(0, 1), n)
    e = rand(Normal(0, 1), n)
    Y0 = beta * Z .+ e
    EST = Array{Float64}(undef, nₐ)
    ESTZ = Array{Float64}(undef, nₐ)
    P = Array{Float64}(undef, nₐ)
    PZ = Array{Float64}(undef, nₐ)
    Δ = Array{Float64}(undef, nₐ)
    SUM = Array{Float64}(undef, nₐ)
    for i = 1:nₐ
        W = 𝒲[i]
        Δ[i] = mean(Z[indnr[i]]) - mean(Z[setdiff(1:n, indnr[i])])
        ols = GLM.fit(LinearModel, [W ones(n)], reshape(Y0, n))
        EST[i] = coef(ols)[1]
        P[i] = coeftable(ols).cols[4][1]
        olsz = GLM.fit(LinearModel, [W Z ones(n)], reshape(Y0, n))
        ESTZ[i] = coef(olsz)[1]
        PZ[i] = coeftable(olsz).cols[4][1]
        SUM[i] = sum(Z[indnr[i]])
    end
    map1 = hcat(sortperm(-abs.(EST)), 1:nₐ)
    map2 = sortslices(map1, dims=1)
    PF = map2[:, 2] ./ nₐ
    return [EST ESTZ P PZ Δ P.<=0.05 PZ.<=0.05 PF PF .<= 0.05]
end

function print_time(rep_nr, tot_reps)
    # Function that return expected time until simulation is finished
    elapsed_time = time_ns() - START_TIME
    perc_compl = (rep_nr / tot_reps) * 100
    exp_tot_time = elapsed_time / (perc_compl / 100)
    eft = Dates.now() + Dates.Nanosecond(floor(exp_tot_time - elapsed_time))
    perc_compl = round(perc_compl, sigdigits=3)
    println(perc_compl, "% completed, expect to finish ", eft)
end

function create_df(out, i)
    # Cut result from the sim_reg()-function into 100 equal-sized groups
    # based on covariate balance (Δ)
    df = DataFrame(out, :auto)
    rename!(df, Symbol.(df_NAMES))
    df[!, :cut] = cut(df[!, :Delta], 100)
    df[!, :OLS_est_sq] = df[!, :OLS_est].^2
    df[!, :DM_est_sq] = df[!, :DM_est].^2
    df_mean = combine(df, :DM_est => mean, :OLS_est => mean,
                          :s_DM => mean, :s_OLS => mean,
                          :DM_est => var, :OLS_est => var,
                          :DM_est_sq => mean, :OLS_est_sq => mean)
    rename!(df_mean, Symbol.(df_mean_NAMES))
    gdf = groupby(df, :cut)
    df_new = combine(gdf, :OLS_est => mean, :DM_est => mean, :p_DM => mean,
                          :p_OLS => mean, :s_DM => mean, :s_OLS => mean,
                          :Delta => minimum, :Delta => mean, :Delta => maximum,
                          :OLS_est => var, :DM_est => var, :p_F => mean,
                          :s_F => mean, :OLS_est_sq => mean,
                          :DM_est_sq => mean, nrow)

    df_new[!, :index] = 1:100
    select!(df_new, Not(:cut))
    df_new[!, :iter] .= i
    return df_new, df_mean
end

function gen_percentiles(df, df_names)
    # Generate averages for each percentile of Δ over repeated samples
    gdf = groupby(df, :index)
    df_agg = combine(gdf, :DM_est_mean => mean, :OLS_est_mean => mean,
                          :p_DM_mean => mean, :p_OLS_mean => mean,
                          :Delta_mean => mean,
                          :s_DM_mean => mean, :s_OLS_mean => mean,
                          :nrow => mean, :DM_est_var => mean, :OLS_est_var => mean,
                          :p_F_mean => mean, :s_F_mean => mean,
                          :DM_est_sq_mean => mean, :OLS_est_sq_mean => mean)
    rename!(df_agg, Symbol.(df_names))
    return df_agg
end

Random.seed!(12345)

n = 20 # Sample size
rsq = 50 # R-squared from a regression of Y(0) on Z
n₁ = Int(n / 2)
n₀ = n - n₁

nₐ = binomial(n, n₁)
indnr = collect(combinations(1:n, n₁))
𝒲 = gen_𝒲()

df_NAMES = ["DM_est", "OLS_est", "p_DM", "p_OLS", "Delta", "s_DM", "s_OLS",
            "p_F", "s_F"]
df_mean_NAMES = ["DM_est", "OLS_est", "s_DM", "s_OLS",
                 "DM_var", "OLS_var", "DM_mse", "OLS_mse"]
df_agg_NAMES = ["percentile", "DM_est", "OLS_est", "p_DM", "p_OLS", "Delta",
                "s_DM", "s_OLS", "nrows", "DM_var", "OLS_var", "p_F", "s_F",
                "DM_mse", "OLS_mse"]

FILENAME_mean_single = "np_complete_mean_single.csv"
FILENAME_agg_single = "np_complete_agg_single.csv"
FILENAME_mean = "np_complete_mean.csv"
FILENAME_agg = "np_complete_agg.csv"

df_agg_single, df_mean_single = create_df(sim_reg(), 1)
df_agg_single = gen_percentiles(df_agg_single, df_agg_NAMES)

REPS = 1000
df = DataFrame()
df_mean = DataFrame(zeros(1, length(df_mean_NAMES)), :auto)
rename!(df_mean, Symbol.(df_mean_NAMES))

START_TIME = time_ns()

for i = 1:REPS
    # Simulate over repeated samples
    df1, df2 = create_df(sim_reg(), i)
    append!(df, df1)
    global df_mean .+= df2
    print_time(i, REPS)
end

df_mean ./= REPS
df_agg = gen_percentiles(df, df_agg_NAMES)

CSV.write(FILENAME_mean_single, df_mean_single)
CSV.write(FILENAME_agg_single, df_agg_single)
CSV.write(FILENAME_mean, df_mean)
CSV.write(FILENAME_agg, df_agg)

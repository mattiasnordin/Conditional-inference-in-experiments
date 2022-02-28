# This code has been tested with Julia version 1.5.1
using LinearAlgebra
using Distributions # Tested with version 0.23.8
using DataFrames # Tested with version 1.2.2
using CSV # Tested with version 0.9.9
using Random
using Dates
using GLM # Tested with version 1.5.1
using MultivariateStats # Tested with version 0.8.0
using CovarianceMatrices # Tested with version 0.9.1
using RandomCorrelationMatrices # Tested with version 1.0.0
using CategoricalArrays # Tested with version 0.10.0

function gen_y_obs(c_ind, y0, y1)
    # Generate y based on the potential outcomes and a vector of indexes for
    # the control group, c_ind
    y = deepcopy(y1)
    y[c_ind] .= y0[c_ind]
    return y
end

function calc_mahal(z0, z1, vi, n, one0, one1)
    # Calculate Mahalanobis distance
    z0_mean = transpose(z0) * one0
    z1_mean = transpose(z1) * one1
    diff_z = z1_mean - z0_mean
    m = dot(diff_z, vi * diff_z)
    return n / 4 * m
end

function robust_ols(x, y, var_pos, sate, power)
    # Calculate the OLS estimate together with p-value for the
    # Eicker-Huber-White covariance matrix. sate is the sample average
    # treatment effect and var_pos is the location of the treatment vector in
    # the x matrix. power=false means testing the null of the treatment effect
    # equaling sate (for size), whereas power=true means testing the null of
    # the treatment effect equaling 0.
    ols = GLM.fit(LinearModel, x, y)
    est = coef(ols)[var_pos]
    try
        se = sqrt(vcov(ols, HC1())[var_pos,var_pos])
        if power == true
            t_stat = abs(est / se)
        else
            t_stat = abs((est - sate) / se)
        end
        p = ccdf(TDist(size(x)[1] - size(x)[2]), t_stat) * 2
        s = p <= 0.05
        return est, p, s
    catch
        p = NaN
        s = NaN
        return est, p, s
    end
end

function reg_ols(x, y, var_pos)
    # Point estimate and p-value for the regular OLS estimator with
    # var_pos being the location of the treatment vector in the x matrix.
    ols = GLM.fit(LinearModel, x, y)
    est = coef(ols)[var_pos]
    p = coeftable(ols).cols[4][var_pos]
    s = p <= 0.05
    return est, p, s
end

function component_selection(deltabar, h, c_ind, vi_list, PZpc,
                             Zpc, n, n0)
    # Algorithm 1 in the paper
    nₐ = binomial(big(n), big(n0))
    ndeltabar = deepcopy(nₐ)
    Zpc0 = Zpc[c_ind, :]
    Zpc1 = Zpc[setdiff(1:n, c_ind), :]
    div_list = ones(n0) / n0
    p = 0
    while (ndeltabar >= h) & (p < PZpc)
        m = calc_mahal(Zpc0[:, 1:p+1], Zpc1[:, 1:p+1], vi_list[p+1],
                       n, div_list, div_list)
        ndeltabar = round(cdf(NoncentralChisq(p + 1, m), deltabar) * nₐ)
        if ndeltabar >= h
            p += 1
        end
    end
    return p
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

function sim_np(samples, corr, hetero, K, n, n0, δ, H, τ, power)
    # Randomly draw a sample and calculate the difference-in-means estimate,
    # the OLS estimate and our estimate based on principal components for
    # samples number of treatment assignments.
    MV = zeros(K)
    Iₖ = Matrix{Float64}(I, K, K)

    if corr == false
        beta = ones(K) / sqrt(K)
        A = Matrix{Float64}(I, K, K)
    elseif corr == true
        beta = sqrt(K) * ones(K)./ K
        A = RandomCorrelationMatrices.randcormatrix(K, 1)
    end
    if K > 1
        Zdist = MvNormal(MV, A)
    elseif K == 1
        Zdist = MvNormal(MV, 1)
    end
    Z = transpose(rand(Zdist, n))
    Zt = Z .- mean(Z, dims=1)
    e = rand(Normal(0, 1), n)
    Y0 = Z * beta .+ e

    if hetero == false
        Y1 = Y0 .+ τ
    elseif hetero == true
        Y1 = Y0 .+ τ .+ rand(Normal(0, 1), n)
    end

    M = fit(PCA, transpose(Z); maxoutdim=K)
    ZPCA = transpose(MultivariateStats.transform(M, transpose(Z)))
    P = size(ZPCA)[2]
    VI_list = [inv(cov(ZPCA[:, 1:p])) for p = 1:P]
    VI_Z = inv(cov(Z))
    sate = mean(Y1) - mean(Y0)
    div_list = ones(n0) / n0

    OUT = Array{Float64}(undef, samples, 12)

    for i = 1:samples
        c_ind = shuffle(1:n)[1:n0]
        t_ind = setdiff(1:n, c_ind)
        yobs = reshape(gen_y_obs(c_ind, Y0, Y1), n)
        W = zeros(n)
        W[t_ind] .= 1
        m = calc_mahal(Z[c_ind, :], Z[setdiff(1:n, c_ind), :],
                       VI_Z, n, div_list, div_list)
        p = component_selection(δ, H, c_ind, VI_list, P, ZPCA, n, n0)
        if hetero == true
            e1, p1, s1 = robust_ols([W ones(n)], yobs, 1, sate, power)
            e2, p2, s2 = robust_ols([W Zt Zt.*W ones(n)], yobs, 1, sate, power)
            e3, p3, s3 = robust_ols(
                [W ZPCA[:, 1:p] ZPCA[:, 1:p].*W ones(n)], yobs, 1, sate, power)
        elseif hetero == false
            e1, p1, s1 = reg_ols([W ones(n)], yobs, 1)
            e2, p2, s2 = reg_ols([W Z ones(n)], yobs, 1)
            e3, p3, s3 = reg_ols([W ZPCA[:, 1:p] ones(n)], yobs, 1)
        end
        OUT[i, :] = [e1, e2, e3, p1, p2, p3, s1, s2, s3, m, p, sate]
    end
    df = DataFrame(OUT, :auto)
    rename!(df, Symbol.(["DM_est", "OLS_est", "ALG_est", "p_DM", "p_OLS",
                        "p_ALG",  "s_DM", "s_OLS", "s_ALG", "M", "NR_P",
                        "SATE"]))

    return df
end

function sim_alg_reps(reps, reps_in_sample, corr, hetero, k, n, n0, δ, h, τ,
                      quantiles, power)
    # Run the sim_np() function over reps random samples. Then aggregate over
    # quantiles of the Mahalanobis distance.
    global START_TIME = time_ns()
    df_all = DataFrame()
    for i = 1:reps
        df = sim_np(reps_in_sample, corr, hetero, k, n, n0, δ, h, τ, power)
        df[!, :cut] .= cut(df[!, :M], quantiles)
        df[!, :OLS_mse] .= (df[!, :OLS_est] .- df[!, :SATE]).^2
        df[!, :DM_mse] .= (df[!, :DM_est] .- df[!, :SATE]).^2
        df[!, :ALG_mse] .= (df[!, :ALG_est] .- df[!, :SATE]).^2
        gdf = groupby(df, :cut)
        df_agg = combine(gdf,
            :OLS_est => mean, :DM_est => mean, :ALG_est => mean,
            :p_OLS => mean, :p_DM => mean, :p_ALG => mean,
            :s_OLS => mean, :s_DM => mean, :s_ALG => mean,
            :M => mean, :NR_P => mean, :SATE => mean,
            :OLS_mse => mean, :DM_mse => mean, :ALG_mse => mean)
        sort!(df_agg, :cut)
        df_agg[!, :quantile] .= 1:quantiles
        select!(df_agg, Not(:cut))
        append!(df_all, df_agg)
        print_time(i, reps)
    end
    gdf_all = groupby(df_all, :quantile)
    df_agg_all = combine(gdf_all,
        :OLS_est_mean => mean, :DM_est_mean => mean, :ALG_est_mean => mean,
        :p_OLS_mean => mean, :p_DM_mean => mean, :p_ALG_mean => mean,
        :s_OLS_mean => mean, :s_DM_mean => mean, :s_ALG_mean => mean,
        :M_mean => mean, :NR_P_mean => mean, :SATE_mean => mean,
        :OLS_mse_mean => mean, :DM_mse_mean => mean, :ALG_mse_mean => mean)
    df_agg_all[!, :corr] .= corr
    df_agg_all[!, :hetero] .= hetero
    df_agg_all[!, :reps_in_sample] .= reps_in_sample
    df_agg_all[!, :reps] .= reps
    df_agg_all[!, :delta_bar] .= δ
    df_agg_all[!, :h] .= h
    df_agg_all[!, :n] .= n
    df_agg_all[!, :n0] .= n0
    df_agg_all[!, :k] .= k
    df_agg_all[!, :tau] .= τ

    df_names = [replace(i, "_mean_mean" => "") for i = names(df_agg_all)]
    rename!(df_agg_all, Symbol.(df_names))

    return df_agg_all
end

N = 50
N0 = Int(N / 2)
H = 100
delta = 0.01
REPS_IN_SAMPLE = 10000
QUANTILES = 100
REPS = 1000
FILENAME = "sim_np.csv"

# The code is run for different combinations of tau, K, CORR and HETERO
for tau = [0.0, 1.0]
for K = [10, 20, 30]
for CORR = [false, true]
HETERO = false

if tau != 0.0
    global POWER = true
else
    global POWER = false
end

df = sim_alg_reps(REPS, REPS_IN_SAMPLE, CORR, HETERO, K,
                  N, N0, delta, H, tau, QUANTILES, POWER)
if isfile(FILENAME) == false
    CSV.write(FILENAME, df)
else
    CSV.write(FILENAME, df, append=true)
end
end
end
end

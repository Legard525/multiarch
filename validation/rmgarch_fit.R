# Reference fits with rmgarch on dji30ret. Writes one CSV of coefficients per
# case plus conditional-correlation paths, forecasts and the DCC test.
suppressMessages(library(rmgarch))
out <- commandArgs(trailingOnly = TRUE)[1]
data(dji30ret)
X10 <- 100 * as.matrix(dji30ret[, 1:10])
write.csv(data.frame(date = rownames(dji30ret), X10), file.path(out, "dji10.csv"), row.names = FALSE)
X5 <- X10[, 1:5]

uspec <- function(model = "sGARCH", dist = "norm", mean = TRUE)
  ugarchspec(mean.model = list(armaOrder = c(0, 0), include.mean = mean),
             variance.model = list(model = model, garchOrder = c(1, 1)),
             distribution.model = dist)

save_fit <- function(name, f, secs, R, extra = NULL) {
  mc <- f@mfit$matcoef
  write.csv(data.frame(name = rownames(mc), coef = mc[, 1], se = mc[, 2],
                       loglik = likelihood(f), secs = secs),
            file.path(out, paste0("coef_", name, ".csv")), row.names = FALSE)
  n <- dim(R)[1]
  idx <- which(upper.tri(diag(n)), arr.ind = TRUE)
  rr <- sapply(seq_len(nrow(idx)), function(k) R[idx[k, 1], idx[k, 2], ])
  write.csv(rr, file.path(out, paste0("rcor_", name, ".csv")), row.names = FALSE)
}

dcc_case <- function(name, X, model = "sGARCH", udist = "norm", mdist = "mvnorm",
                     dcc = "DCC", mean = TRUE) {
  us <- uspec(model, udist, mean)
  sp <- dccspec(multispec(replicate(ncol(X), us)), dccOrder = c(1, 1),
                model = dcc, distribution = mdist)
  t0 <- Sys.time(); f <- dccfit(sp, data = X, solver = "solnp")
  save_fit(name, f, as.numeric(difftime(Sys.time(), t0, units = "secs")), rcor(f))
  f
}

f <- dcc_case("dcc_norm", X5)
fc <- dccforecast(f, n.ahead = 10)
H <- rcov(fc)[[1]]
write.csv(t(sapply(1:10, function(h) H[, , h][upper.tri(diag(5), diag = TRUE)])),
          file.path(out, "forecast_dcc_norm.csv"), row.names = FALSE)
dcc_case("adcc_norm", X5, dcc = "aDCC")
dcc_case("dcc_t", X5, udist = "std", mdist = "mvt")
dcc_case("dcc_gjr", X5, model = "gjrGARCH")
dcc_case("dcc_zero", X5, mean = FALSE)
dcc_case("dcc_n10", X10)
write.csv(data.frame(X5[, 1:2] / 100), file.path(out, "raw2.csv"), row.names = FALSE)
dcc_case("dcc_raw2", X5[, 1:2] / 100)

cop_case <- function(name, X, udist, copula) {
  sp <- cgarchspec(multispec(replicate(ncol(X), uspec("sGARCH", udist))), dccOrder = c(1, 1),
                   distribution.model = list(copula = copula, method = "ML",
                                             time.varying = TRUE, transformation = "parametric"))
  t0 <- Sys.time(); f <- cgarchfit(sp, data = X, solver = "solnp")
  save_fit(name, f, as.numeric(difftime(Sys.time(), t0, units = "secs")), rcor(f))
}
cop_case("cop_t", X5, "std", "mvt")
cop_case("cop_norm", X5, "std", "mvnorm")

gs <- gogarchspec(mean.model = list(model = "constant"), variance.model = list(model = "sGARCH"),
                  distribution.model = "mvnorm", ica = "fastica")
set.seed(1)
t0 <- Sys.time(); g <- gogarchfit(gs, data = X5)
gsecs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.csv(data.frame(loglik = likelihood(g), secs = gsecs), file.path(out, "gogarch.csv"), row.names = FALSE)
write.csv(g@mfit$W, file.path(out, "gogarch_W.csv"), row.names = FALSE)
write.csv(coef(g), file.path(out, "gogarch_coef.csv"))

# DCCtest on the residuals rmgarch itself uses (whitened, from a static copula)
cspec <- cgarchspec(multispec(replicate(5, uspec())),
                    distribution.model = list(copula = "mvnorm", transformation = "parametric",
                                              method = "Kendall"))
cfit <- cgarchfit(cspec, X5, fit.control = list(eval.se = FALSE))
Z <- cfit@mfit$stdresid
write.csv(Z, file.path(out, "dcctest_Z.csv"), row.names = FALSE)
dt <- DCCtest(X5, n.lags = 5, Z = Z)
write.csv(data.frame(stat = dt$statistic, p = dt$p.value), file.path(out, "dcctest.csv"), row.names = FALSE)

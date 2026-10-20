# Course workflow (FRF notebook ch. 22) with tsmarch: GARCH(1,1) margins with
# a constant, then DCC / aDCC / CGARCH. Raw (fractional) returns, as in the notes.
suppressMessages({library(tsmarch); library(tsgarch); library(xts)})
out <- commandArgs(trailingOnly = TRUE)[1]
X <- read.csv(file.path(out, "dji5.csv"), row.names = 1) / 100
y <- xts(as.matrix(X), order.by = as.Date(rownames(X)))
fit_margins <- function(y) {
  m <- lapply(1:ncol(y), function(i)
    estimate(garch_modelspec(y[, i], model = "garch", order = c(1, 1), constant = TRUE), keep_tmb = TRUE))
  names(m) <- colnames(y)
  to_multi_estimate(m)
}
cases <- list(
  ts_dcc2 = list(cols = 1:2, dyn = "dcc", dist = "mvn"),
  ts_dcc5 = list(cols = 1:5, dyn = "dcc", dist = "mvn"),
  ts_adcc5 = list(cols = 1:5, dyn = "adcc", dist = "mvn"),
  ts_dcct5 = list(cols = 1:5, dyn = "dcc", dist = "mvt"))
for (nm in names(cases)) {
  cs <- cases[[nm]]
  mg <- fit_margins(y[, cs$cols])
  t0 <- Sys.time()
  spec <- dcc_modelspec(mg, dynamics = cs$dyn, dcc_order = c(1, 1), distribution = cs$dist)
  f <- estimate(spec)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  up <- do.call(rbind, lapply(names(mg), function(s) {
    cf <- coef(mg[[s]]); data.frame(name = paste0("[", s, "].", names(cf)), coef = unname(cf))
  }))
  cc <- coef(f)
  write.csv(rbind(up, data.frame(name = paste0("[Joint]", names(cc)), coef = unname(cc))),
            file.path(out, paste0("ts_coef_", nm, ".csv")), row.names = FALSE)
  H <- tscov(f)
  R <- tscor(f)
  write.csv(data.frame(loglik = as.numeric(logLik(f)), secs = secs, rho12 = R[1, 2, ]),
            file.path(out, paste0("ts_", nm, ".csv")), row.names = FALSE)
}

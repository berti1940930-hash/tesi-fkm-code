###########################################################################
###########################################################################
###########   TESI – Analisi Costi di Transazione (TC)           #########
###########   Strategia FKM Intra-Cluster Reversal               #########
###########################################################################
###########################################################################
#
#  Replica la metodologia di Vitale (2025), Sezione 3.5 e Tabella 6.
#  Per ciascuna configurazione (C × m) e soglia di membership (tau):
#
#    Scenario 0 bps  : rendimenti lordi (gross)
#    Scenario 10 bps : rendimenti netti con TC = turnover × 10bps
#    Scenario 20 bps : rendimenti netti con TC = turnover × 20bps
#
#  Il turnover è calcolato mese per mese come:
#    turnover_t = 0.5 × Σ|w_t – w_{t-1}|
#  dove w_t è il vettore dei pesi del portafoglio al mese t
#  (lunghi +1/n_long, corti −1/n_short, zero altrimenti).
#
#  PREREQUISITI
#    – final_df.rds           (panel 285 titoli, 2004-02..2020-12)
#    – refit_10configs_RS20.rds  (FKM fits mensili, 10 configurazioni)
#  entrambi nella cartella INPUT_DIR.
#
#  OUTPUT (in OUT_DIR)
#    – TC_monthly_all.rds     : rendimenti mensili gross/net per ogni
#                               config × tau × cost
#    – TC_summary_all.csv     : tabella riassuntiva completa
#    – TC_table_best.csv      : migliori config + reversal, stile Table 6
#    – TC_table_best_latex.txt: versione LaTeX della tabella best
#
###########################################################################

library(dplyr)
library(data.table)

###########################################################################
# 0. CONFIGURAZIONE
###########################################################################

INPUT_DIR <- "/Users/macbookpro13/Desktop/TESI copy/05_Outputs"
OUT_DIR   <- INPUT_DIR   # salva nella stessa cartella
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

TAU_GRID  <- c(0, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
COST_BPS  <- c(0, 10, 20)
N_BEST    <- 8L

LEGACY_RETURNS_CANDIDATES <- c(
  file.path(INPUT_DIR, "FKM", "Rendimenti_Strategia_Tau.rds"),
  file.path(INPUT_DIR, "strategy_returns_tau.rds")
)

first_existing_path <- function(paths) {
  hits <- paths[file.exists(paths)]
  if (length(hits) == 0) return(NA_character_)
  hits[[1]]
}

format_tau_label <- function(x) {
  if (isTRUE(all.equal(x, 0))) return("0")
  txt <- format(x, nsmall = 2, scientific = FALSE, trim = TRUE)
  txt <- sub("0+$", "", txt)
  sub("\\.$", "", txt)
}

cat("============================================================\n")
cat("  TC Analysis – FKM Intra-Cluster Reversal Strategy\n")
cat("  Costi testati:", paste0(COST_BPS, "bps", collapse = ", "), "\n")
cat("  Tau grid:     ", paste(TAU_GRID, collapse = ", "), "\n")
cat("============================================================\n\n")

###########################################################################
# 1. CARICAMENTO DATI
###########################################################################

cat("[1/6] Caricamento final_df e refit_10configs_RS20...\n")

final_df <- readRDS(file.path(INPUT_DIR, "final_df.rds"))
if (!"ym" %in% names(final_df))
  final_df <- final_df %>% mutate(ym = format(date, "%Y-%m"))
final_df$ym     <- as.character(final_df$ym)
final_df$permno <- as.character(final_df$permno)

refit_all <- readRDS(file.path(INPUT_DIR, "refit_10configs_RS20.rds"))
config_names <- names(refit_all)
months_sorted <- sort(unique(final_df$ym))

legacy_returns_path <- first_existing_path(LEGACY_RETURNS_CANDIDATES)
legacy_returns <- if (!is.na(legacy_returns_path)) readRDS(legacy_returns_path) else NULL

cat("  Titoli:         ", length(unique(final_df$permno)), "\n")
cat("  Mesi:           ", length(months_sorted), "\n")
cat("  Configurazioni: ", length(config_names), "\n\n")
if (is.null(legacy_returns)) {
  cat("  Rendimenti storici: nessun file trovato, usero il lordo ricalcolato.\n\n")
} else {
  cat("  Rendimenti storici:", legacy_returns_path, "\n")
  cat("  Serie disponibili: ", length(legacy_returns), "\n\n")
}

###########################################################################
# 2. FUNZIONI CORE
###########################################################################

# --- Vettore dei pesi portafoglio ---
build_weight_vector <- function(long_ids, short_ids) {
  weights <- numeric(0)
  if (length(long_ids) > 0) {
    lw <- rep(1 / length(long_ids), length(long_ids))
    names(lw) <- as.character(long_ids)
    weights <- lw
  }
  if (length(short_ids) > 0) {
    sw <- rep(-1 / length(short_ids), length(short_ids))
    names(sw) <- as.character(short_ids)
    overlap <- intersect(names(weights), names(sw))
    if (length(overlap) > 0) weights[overlap] <- weights[overlap] + sw[overlap]
    fresh <- setdiff(names(sw), names(weights))
    if (length(fresh) > 0) weights <- c(weights, sw[fresh])
  }
  weights
}

# --- Turnover: 0.5 × Σ|w_t – w_{t-1}| ---
calc_turnover <- function(prev_w, cur_w) {
  ids <- union(names(prev_w), names(cur_w))
  if (length(ids) == 0) return(0)
  prev <- setNames(numeric(length(ids)), ids)
  cur  <- setNames(numeric(length(ids)), ids)
  if (length(prev_w) > 0) prev[names(prev_w)] <- prev_w
  if (length(cur_w)  > 0) cur[names(cur_w)]   <- cur_w
  0.5 * sum(abs(cur - prev))
}

build_tau_key <- function(config, tau) {
  paste0(config, "_tau", format_tau_label(tau))
}

overlay_precomputed_gross <- function(monthly_tc, precomputed_df) {
  if (is.null(precomputed_df) || nrow(monthly_tc) == 0) return(monthly_tc)
  if (!all(c("ym", "ret") %in% names(precomputed_df))) return(monthly_tc)

  idx <- match(monthly_tc$ym, as.character(precomputed_df$ym))
  if (anyNA(idx)) {
    missing_ym <- paste(monthly_tc$ym[is.na(idx)], collapse = ", ")
    warning("Mesi mancanti nei rendimenti precalcolati: ", missing_ym)
    return(monthly_tc)
  }

  monthly_adj <- monthly_tc
  pre_aligned <- precomputed_df[idx, , drop = FALSE]

  monthly_adj$gross_ret <- as.numeric(pre_aligned$ret)
  monthly_adj$net_ret   <- monthly_adj$gross_ret - monthly_adj$cost

  for (nm in c("n_pairs_pre", "n_pairs_post",
               "n_stocks_kept", "n_stocks_filtered")) {
    if (nm %in% names(pre_aligned)) {
      monthly_adj[[nm]] <- as.numeric(pre_aligned[[nm]])
    }
  }

  monthly_adj$cum_gross <- cumprod(1 + monthly_adj$gross_ret)
  monthly_adj$cum_net   <- cumprod(1 + monthly_adj$net_ret)
  monthly_adj
}

# --- Strategia mensile con TC ---
run_strategy_tc <- function(config_fits, final_df, months_sorted,
                            tau = 0, cost_bps = 10,
                            liquidate_at_end = TRUE) {
  cost_rate    <- cost_bps / 10000
  prev_weights <- numeric(0)
  out          <- vector("list", max(0L, length(months_sorted) - 1L))

  move_to_cash <- function(row, prev_w, cost_rate) {
    to <- calc_turnover(prev_w, numeric(0))
    row$turnover <- to
    row$cost     <- to * cost_rate
    row$net_ret  <- -row$cost
    row
  }

  for (i in seq_len(length(months_sorted) - 1L)) {
    t_month  <- months_sorted[i]
    t1_month <- months_sorted[i + 1L]
    fit_t    <- config_fits[[t_month]]

    empty <- list(ym = t_month, gross_ret = 0, net_ret = 0,
                  turnover = 0, cost = 0,
                  n_pairs_pre = 0, n_pairs_post = 0,
                  n_stocks_kept = 0, n_stocks_filtered = 0,
                  n_long = 0, n_short = 0)

    if (is.null(fit_t)) {
      out[[i]] <- move_to_cash(empty, prev_weights, cost_rate)
      prev_weights <- numeric(0)
      next
    }

    sig_t <- final_df[final_df$ym == t_month,  c("permno", "ret")]
    names(sig_t)[2] <- "signal"
    pay_t1 <- final_df[final_df$ym == t1_month, c("permno", "ret")]
    names(pay_t1)[2] <- "ret_fwd"

    u_max <- if (!is.null(fit_t$u_max)) {
      fit_t$u_max
    } else {
      apply(fit_t$U, 1, max)
    }

    cl_df <- data.frame(permno  = as.character(fit_t$permno),
                        cluster = fit_t$assign,
                        u_max   = u_max,
                        stringsAsFactors = FALSE)

    trade_df <- merge(cl_df, sig_t,  by = "permno")
    trade_df <- merge(trade_df, pay_t1, by = "permno")
    trade_df <- trade_df[is.finite(trade_df$signal) &
                         is.finite(trade_df$ret_fwd), , drop = FALSE]

    n_total <- nrow(trade_df)
    if (n_total == 0) {
      out[[i]] <- move_to_cash(empty, prev_weights, cost_rate)
      prev_weights <- numeric(0); next
    }

    if (tau > 0) trade_df <- trade_df[trade_df$u_max >= tau, , drop = FALSE]
    n_kept     <- nrow(trade_df)
    n_filtered <- n_total - n_kept

    if (n_kept < 2) {
      empty$n_stocks_kept     <- n_kept
      empty$n_stocks_filtered <- n_filtered
      out[[i]] <- move_to_cash(empty, prev_weights, cost_rate)
      prev_weights <- numeric(0); next
    }

    clusters <- sort(unique(trade_df$cluster))
    all_diffs  <- numeric(0)
    pair_store <- list()

    for (cl in clusters) {
      sub     <- trade_df[trade_df$cluster == cl, , drop = FALSE]
      sub     <- sub[order(sub$signal), , drop = FALSE]
      n_cl    <- nrow(sub)
      n_pairs <- floor(n_cl / 2)
      if (n_pairs < 1) next

      lo  <- seq_len(n_pairs)
      hi  <- rev(seq.int(n_cl - n_pairs + 1L, n_cl))
      diffs <- sub$signal[hi] - sub$signal[lo]
      all_diffs <- c(all_diffs, diffs)

      pair_store[[as.character(cl)]] <- list(
        long_id  = sub$permno[lo],
        short_id = sub$permno[hi],
        long_ret = sub$ret_fwd[lo],
        short_ret = sub$ret_fwd[hi],
        diffs    = diffs
      )
    }

    if (length(all_diffs) == 0) {
      empty$n_stocks_kept <- n_kept; empty$n_stocks_filtered <- n_filtered
      out[[i]] <- move_to_cash(empty, prev_weights, cost_rate)
      prev_weights <- numeric(0); next
    }

    sigma <- sd(all_diffs, na.rm = TRUE)
    if (!is.finite(sigma) || sigma <= 0) {
      empty$n_pairs_pre <- length(all_diffs)
      empty$n_stocks_kept <- n_kept; empty$n_stocks_filtered <- n_filtered
      out[[i]] <- move_to_cash(empty, prev_weights, cost_rate)
      prev_weights <- numeric(0); next
    }

    pair_rets  <- numeric(0)
    long_ids   <- character(0)
    short_ids  <- character(0)

    for (cl in names(pair_store)) {
      p    <- pair_store[[cl]]
      keep <- which(p$diffs > sigma)
      if (length(keep) == 0) next
      pair_rets <- c(pair_rets, p$long_ret[keep] - p$short_ret[keep])
      long_ids  <- c(long_ids,  as.character(p$long_id[keep]))
      short_ids <- c(short_ids, as.character(p$short_id[keep]))
    }

    cur_weights <- build_weight_vector(long_ids, short_ids)
    to          <- calc_turnover(prev_weights, cur_weights)
    cost        <- to * cost_rate
    gross_ret   <- if (length(pair_rets) == 0) 0 else mean(pair_rets)
    net_ret     <- gross_ret - cost

    out[[i]] <- list(
      ym = t_month, gross_ret = gross_ret, net_ret = net_ret,
      turnover = to, cost = cost,
      n_pairs_pre  = length(all_diffs),
      n_pairs_post = length(pair_rets),
      n_stocks_kept    = n_kept,
      n_stocks_filtered = n_filtered,
      n_long  = length(long_ids),
      n_short = length(short_ids)
    )
    prev_weights <- cur_weights
  }

  df <- do.call(rbind.data.frame, c(out, stringsAsFactors = FALSE))
  for (col in c("gross_ret","net_ret","turnover","cost","n_pairs_pre",
                "n_pairs_post","n_stocks_kept","n_stocks_filtered",
                "n_long","n_short"))
    df[[col]] <- as.numeric(df[[col]])

  # Liquidazione finale
  if (liquidate_at_end && nrow(df) > 0 && length(prev_weights) > 0) {
    liq_cost <- calc_turnover(prev_weights, numeric(0)) * cost_rate
    df$cost[nrow(df)]    <- df$cost[nrow(df)] + liq_cost
    df$net_ret[nrow(df)] <- df$net_ret[nrow(df)] - liq_cost
  }

  df$cum_gross <- cumprod(1 + df$gross_ret)
  df$cum_net   <- cumprod(1 + df$net_ret)
  df
}

# --- Metriche di performance ---
calc_perf <- function(df, config, tau, cost_bps) {
  gross <- df$gross_ret
  net   <- df$net_ret

  pf <- function(r) {
    r <- r[is.finite(r)]
    ann_r <- mean(r) * 12
    ann_v <- sd(r)   * sqrt(12)
    sh    <- if (ann_v > 0) ann_r / ann_v else NA_real_
    dd_v  <- sqrt(mean(pmin(r, 0)^2)) * sqrt(12)
    so    <- if (dd_v > 0) ann_r / dd_v else NA_real_
    cum   <- cumprod(1 + r)
    mdd   <- min((cum - cummax(cum)) / cummax(cum))
    cal   <- if (mdd != 0) ann_r / abs(mdd) else NA_real_
    c(ann_r, ann_v, sh, so, mdd, cal, prod(1 + r) - 1, length(r))
  }

  g <- pf(gross); n <- pf(net)

  avg_to   <- mean(df$turnover, na.rm = TRUE)
  avg_cost <- mean(df$cost,     na.rm = TRUE)
  pct_0    <- mean(df$n_pairs_post == 0) * 100
  avg_pair <- mean(df$n_pairs_post, na.rm = TRUE)

  data.frame(
    config           = config,
    tau              = tau,
    cost_bps         = cost_bps,
    n_months         = g[8],
    # Gross
    gross_ret_pct    = round(g[1] * 100, 4),
    gross_vol_pct    = round(g[2] * 100, 4),
    gross_sharpe     = round(g[3],       4),
    gross_sortino    = round(g[4],       4),
    gross_maxdd_pct  = round(g[5] * 100, 2),
    gross_calmar     = round(g[6],       4),
    gross_tot_pct    = round(g[7] * 100, 2),
    # Net
    net_ret_pct      = round(n[1] * 100, 4),
    net_vol_pct      = round(n[2] * 100, 4),
    net_sharpe       = round(n[3],       4),
    net_sortino      = round(n[4],       4),
    net_maxdd_pct    = round(n[5] * 100, 2),
    net_calmar       = round(n[6],       4),
    net_tot_pct      = round(n[7] * 100, 2),
    # Costi
    avg_turnover     = round(avg_to,   4),
    avg_cost_pct_m   = round(avg_cost * 100, 4),
    avg_cost_pct_ann = round(avg_cost * 12 * 100, 4),
    pct_no_trade     = round(pct_0, 1),
    avg_pairs        = round(avg_pair, 2),
    stringsAsFactors = FALSE
  )
}

###########################################################################
# 3. REVERSAL BENCHMARK (no cluster assignments needed)
###########################################################################

cat("[2/6] Calcolo Reversal Benchmark...\n")

reversal_gross <- sapply(
  seq_along(months_sorted)[-length(months_sorted)],
  function(ti) {
    dfm_t  <- final_df[final_df$ym == months_sorted[ti],     c("permno","ret")]
    dfm_t1 <- final_df[final_df$ym == months_sorted[ti + 1], c("permno","ret")]
    dfm <- merge(dfm_t, dfm_t1, by = "permno", suffixes = c("_t","_t1"))
    dfm <- dfm[is.finite(dfm$ret_t) & is.finite(dfm$ret_t1), ]
    if (nrow(dfm) < 10) return(NA_real_)
    q <- quantile(dfm$ret_t, probs = c(0.2, 0.8), na.rm = TRUE)
    mean(dfm$ret_t1[dfm$ret_t <= q[1]]) - mean(dfm$ret_t1[dfm$ret_t >= q[2]])
  }
)

rev_df <- data.frame(
  ym        = months_sorted[-length(months_sorted)],
  gross_ret = reversal_gross,
  stringsAsFactors = FALSE
)
rev_df <- rev_df[is.finite(rev_df$gross_ret), ]

# Reversal ha turnover ≈ 100% ogni mese (top/bottom 20% totalmente sostituiti)
# Stima conservativa: 40% dei titoli cambia ogni mese (overlapping percentile)
# → turnover ≈ 0.80 (40% × 2 lati / portafoglio)
# Per semplicità usiamo turnover = 1.0 (worst case) come in letteratura
REV_TURNOVER_EST <- 1.0

rev_rows <- lapply(COST_BPS, function(bps) {
  cost_m  <- REV_TURNOVER_EST * bps / 10000
  net_rets <- rev_df$gross_ret - cost_m
  rev_df$net_ret <- net_rets
  rev_df$turnover <- REV_TURNOVER_EST
  rev_df$cost <- cost_m

  r <- rev_df$net_ret[is.finite(rev_df$net_ret)]
  g <- rev_df$gross_ret[is.finite(rev_df$gross_ret)]
  pf <- function(x) {
    ann_r <- mean(x)*12; ann_v <- sd(x)*sqrt(12)
    sh <- if(ann_v>0) ann_r/ann_v else NA
    dd_v <- sqrt(mean(pmin(x,0)^2))*sqrt(12)
    so <- if(dd_v>0) ann_r/dd_v else NA
    cum <- cumprod(1+x); mdd <- min((cum-cummax(cum))/cummax(cum))
    cal <- if(mdd!=0) ann_r/abs(mdd) else NA
    c(ann_r, ann_v, sh, so, mdd, cal, prod(1+x)-1, length(x))
  }
  gv <- pf(g); nv <- pf(r)
  data.frame(
    config="Reversal_Benchmark", tau=0, cost_bps=bps, n_months=gv[8],
    gross_ret_pct=round(gv[1]*100,4), gross_vol_pct=round(gv[2]*100,4),
    gross_sharpe=round(gv[3],4), gross_sortino=round(gv[4],4),
    gross_maxdd_pct=round(gv[5]*100,2), gross_calmar=round(gv[6],4),
    gross_tot_pct=round(gv[7]*100,2),
    net_ret_pct=round(nv[1]*100,4), net_vol_pct=round(nv[2]*100,4),
    net_sharpe=round(nv[3],4), net_sortino=round(nv[4],4),
    net_maxdd_pct=round(nv[5]*100,2), net_calmar=round(nv[6],4),
    net_tot_pct=round(nv[7]*100,2),
    avg_turnover=REV_TURNOVER_EST, avg_cost_pct_m=cost_m*100,
    avg_cost_pct_ann=cost_m*12*100, pct_no_trade=0, avg_pairs=NA,
    stringsAsFactors=FALSE
  )
})

rev_summary <- do.call(rbind, rev_rows)
cat("  ✓ Reversal calcolato:", nrow(rev_df), "mesi\n\n")

###########################################################################
# 4. LOOP PRINCIPALE: tutti i config × tau × cost
###########################################################################

cat("[3/6] Strategia FKM –", length(config_names), "config ×",
    length(TAU_GRID), "tau ×", length(COST_BPS), "scenari TC\n")

all_summary  <- list()
all_monthly  <- list()
total_runs   <- length(config_names) * length(TAU_GRID)
run_count    <- 0
t0_all       <- Sys.time()

for (cfg in config_names) {
  fits <- refit_all[[cfg]]

  for (tau in TAU_GRID) {
    run_count <- run_count + 1
    pct       <- round(run_count / total_runs * 100)
    cat(sprintf("  [%3d%%] %s / tau=%.2f ... ", pct, cfg, tau))
    tau_key <- build_tau_key(cfg, tau)
    precomputed_monthly <- if (!is.null(legacy_returns)) legacy_returns[[tau_key]] else NULL

    monthly_tc_0 <- tryCatch(
      run_strategy_tc(fits, final_df, months_sorted, tau = tau, cost_bps = 0),
      error = function(e) {
        cat("ERRORE:", conditionMessage(e), "\n"); NULL
      }
    )
    if (is.null(monthly_tc_0)) next

    use_precomputed <- !is.null(precomputed_monthly)
    monthly_0 <- if (use_precomputed) {
      overlay_precomputed_gross(monthly_tc_0, precomputed_monthly)
    } else {
      monthly_tc_0
    }

    for (bps in COST_BPS) {
      # Per bps=0 riutilizziamo monthly_0 già calcolato.
      # Per bps>0 ri-eseguiamo la funzione: è necessario perché il costo di
      # liquidazione finale dipende da cost_bps e non è salvato nella colonna
      # turnover di monthly_0 (che riflette solo il turnover di trading mensile).
      if (bps == 0) {
        monthly_bps <- monthly_0
      } else {
        monthly_bps <- tryCatch(
          run_strategy_tc(fits, final_df, months_sorted,
                          tau = tau, cost_bps = bps,
                          liquidate_at_end = TRUE),
          error = function(e) {
            cat("ERRORE (", bps, "bps):", conditionMessage(e), "\n"); NULL
          }
        )
        if (is.null(monthly_bps)) next
        if (use_precomputed) {
          monthly_bps <- overlay_precomputed_gross(monthly_bps, precomputed_monthly)
        }
      }

      key <- paste0(cfg, "_tau", gsub("\\.", "p", format(tau)), "_", bps, "bps")
      perf_row <- calc_perf(monthly_bps, cfg, tau, bps)
      all_summary[[key]] <- perf_row
      all_monthly[[key]] <- cbind(config = cfg, tau = tau, cost_bps = bps,
                                  monthly_bps, stringsAsFactors = FALSE)
    }
    cat(if (use_precomputed) "✓ [lordo precalcolato]\n" else "✓ [lordo ricalcolato]\n")
  }
}

cat("\n  Completato in", round(difftime(Sys.time(), t0_all, units="mins"),1), "min\n\n")

###########################################################################
# 5. ASSEMBLAGGIO E SALVATAGGIO
###########################################################################

cat("[4/6] Assemblaggio risultati...\n")

summary_fkm <- do.call(rbind, all_summary)
rownames(summary_fkm) <- NULL
summary_all_final <- rbind(summary_fkm, rev_summary)
rownames(summary_all_final) <- NULL

write.csv(summary_all_final,
          file.path(OUT_DIR, "TC_summary_all.csv"), row.names = FALSE)
cat("  ✓ TC_summary_all.csv salvato (", nrow(summary_all_final), "righe )\n")

# Salva monthly
all_monthly_df <- do.call(rbind, all_monthly)
saveRDS(all_monthly_df, file.path(OUT_DIR, "TC_monthly_all.rds"))
cat("  ✓ TC_monthly_all.rds salvato\n\n")

###########################################################################
# 6. TABELLA BEST (stile Tabella 6 di Vitale)
###########################################################################

cat("[5/6] Costruzione tabella best (stile Vitale Table 6)...\n")

# Seleziona le migliori configurazioni per sharpe lordo (da bps=0)
top_labels <- summary_all_final %>%
  filter(cost_bps == 0, config != "Reversal_Benchmark") %>%
  arrange(desc(gross_sharpe)) %>%
  slice_head(n = N_BEST) %>%
  mutate(label = paste0(config, "_tau", tau)) %>%
  pull(label)

cat("  Top", N_BEST, "configurazioni (per Sharpe lordo):\n")
for (lb in top_labels) cat("   –", lb, "\n")

# Costruisci la tabella: per ogni config × cost, una riga
build_best_table <- function(summary_df, top_labels) {
  rows <- list()

  for (lb in top_labels) {
    parts   <- strsplit(lb, "_tau")[[1]]
    cfg     <- parts[1]
    tau_val <- as.numeric(parts[2])

    for (bps in COST_BPS) {
      r <- summary_df %>%
        filter(config == cfg, tau == tau_val, cost_bps == bps)
      if (nrow(r) == 0) next

      ret_col <- if (bps == 0) r$gross_ret_pct else r$net_ret_pct
      sh_col  <- if (bps == 0) r$gross_sharpe   else r$net_sharpe
      dd_col  <- if (bps == 0) r$gross_maxdd_pct else r$net_maxdd_pct

      rows[[length(rows) + 1]] <- data.frame(
        Strategia  = paste0(cfg, " / τ=", tau_val),
        TC_bps     = bps,
        Ret_ann_pct = round(ret_col, 4),
        Sharpe      = round(sh_col, 4),
        MaxDD_pct   = round(dd_col, 2),
        Turnover_med = round(r$avg_turnover, 4),
        stringsAsFactors = FALSE
      )
    }
  }

  # Aggiungi reversal
  for (bps in COST_BPS) {
    r <- summary_df %>% filter(config == "Reversal_Benchmark", cost_bps == bps)
    if (nrow(r) == 0) next
    ret_col <- if (bps == 0) r$gross_ret_pct else r$net_ret_pct
    sh_col  <- if (bps == 0) r$gross_sharpe   else r$net_sharpe
    dd_col  <- if (bps == 0) r$gross_maxdd_pct else r$net_maxdd_pct
    rows[[length(rows) + 1]] <- data.frame(
      Strategia   = "Reversal",
      TC_bps      = bps,
      Ret_ann_pct = round(ret_col, 4),
      Sharpe      = round(sh_col, 4),
      MaxDD_pct   = round(dd_col, 2),
      Turnover_med = REV_TURNOVER_EST,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, rows)
}

best_table <- build_best_table(summary_all_final, top_labels)
write.csv(best_table, file.path(OUT_DIR, "TC_table_best.csv"), row.names = FALSE)
cat("  ✓ TC_table_best.csv salvato\n")

###########################################################################
# 7. TABELLA LATEX (stile Table 6)
###########################################################################

cat("[6/6] Generazione tabella LaTeX...\n")

make_latex_table <- function(best_table) {
  strategies <- unique(best_table$Strategia)
  bps_vec    <- c(0, 10, 20)

  format_strategy_latex <- function(st) {
    if (identical(st, "Reversal")) return("\\textit{Reversal}")

    parts <- strsplit(st, " / ", fixed = TRUE)[[1]]
    cfg   <- parts[1]
    tau   <- sub("^τ=", "", parts[2])

    cfg_parts <- strsplit(cfg, "_", fixed = TRUE)[[1]]
    c_val <- sub("^C", "", cfg_parts[1])
    m_val <- sub("^m", "", cfg_parts[2])

    sprintf("$C=%s$, $m=%s$, $\\tau=%s$", c_val, m_val, tau)
  }

  header <- paste0(
    "\\begin{table}[H]\n",
    "  \\centering\n",
    "  \\caption{Impatto dei Costi di Transazione sulla Strategia FKM}\n",
    "  \\label{tab:tc_fkm}\n",
    "  \\small\n",
    "  \\begin{tabular}{lrrrrrrrr}\n",
    "  \\toprule\n",
    "  & \\multicolumn{2}{c}{0 bps (lordo)} & ",
    "\\multicolumn{2}{c}{10 bps} & ",
    "\\multicolumn{2}{c}{20 bps} & \\\\\n",
    "  \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n",
    "  Strategia & Ret\\% & Sharpe & Ret\\% & Sharpe & ",
    "Ret\\% & Sharpe & MaxDD\\% & Turn. \\\\\n",
    "  \\midrule\n"
  )

  body_rows <- c()
  for (st in strategies) {
    vals <- list()
    for (bps in bps_vec) {
      r <- best_table[best_table$Strategia == st & best_table$TC_bps == bps, ]
      vals[[as.character(bps)]] <- r
    }
    r0  <- vals[["0"]];  r10 <- vals[["10"]]; r20 <- vals[["20"]]
    if (nrow(r0) == 0 || nrow(r10) == 0 || nrow(r20) == 0) next

    st_label <- format_strategy_latex(st)

    row_str <- sprintf(
      "  %s & %.2f & %.4f & %.2f & %.4f & %.2f & %.4f & %.2f & %.4f \\\\",
      st_label,
      r0$Ret_ann_pct,  r0$Sharpe,
      r10$Ret_ann_pct, r10$Sharpe,
      r20$Ret_ann_pct, r20$Sharpe,
      r0$MaxDD_pct,    r0$Turnover_med
    )
    body_rows <- c(body_rows, row_str)

    # Linea separatrice prima del reversal
    if (st == strategies[length(strategies) - 1]) {
      body_rows <- c(body_rows, "  \\midrule")
    }
  }

  footer <- paste0(
    "\n  \\bottomrule\n",
    "  \\end{tabular}\n",
    "  \\\\[4pt]\n",
    "  {\\footnotesize\n",
    "  Costi di transazione applicati mensile tramite modello di turnover.\n",
    "  Turnover$_t = 0.5\\sum|w_t - w_{t-1}|$, dove $w$ è il vettore\n",
    "  dei pesi (posizioni lunghe $+1/n_L$, corte $-1/n_S$).\n",
    "  Il campione copre 2004-02--2020-12 (202 mesi).\n",
    "  }\n",
    "\\end{table}\n"
  )

  paste0(header, paste(body_rows, collapse = "\n"), footer)
}

latex_str <- make_latex_table(best_table)
latex_path <- file.path(OUT_DIR, "TC_table_best_latex.txt")
writeLines(latex_str, latex_path)
cat("  ✓ TC_table_best_latex.txt salvato\n\n")

###########################################################################
# STAMPA RISULTATI A SCHERMO
###########################################################################

cat("============================================================\n")
cat("  RISULTATI PRINCIPALI\n")
cat("============================================================\n\n")

# Top 5 per Sharpe netto a 10bps
cat("Top 10 per Sharpe NETTO (10 bps):\n")
top10_net <- summary_all_final %>%
  filter(cost_bps == 10, config != "Reversal_Benchmark") %>%
  arrange(desc(net_sharpe)) %>%
  slice_head(n = 10) %>%
  mutate(label = paste0(config, " τ=", tau)) %>%
  select(label, gross_ret_pct, gross_sharpe,
         net_ret_pct, net_sharpe, net_maxdd_pct,
         avg_turnover, avg_cost_pct_ann)

print(top10_net, digits = 4, row.names = FALSE)

cat("\nReversal Benchmark:\n")
rev_print <- rev_summary %>%
  select(cost_bps, gross_ret_pct, gross_sharpe,
         net_ret_pct, net_sharpe, net_maxdd_pct)
print(rev_print, digits = 4, row.names = FALSE)

cat("\nTabella best (stile Vitale Table 6):\n")
print(best_table, row.names = FALSE, digits = 4)

cat("\n============================================================\n")
cat("  File salvati in:", OUT_DIR, "\n")
cat("    - TC_summary_all.csv       (tutti i risultati)\n")
cat("    - TC_monthly_all.rds       (serie mensili)\n")
cat("    - TC_table_best.csv        (tabella best, stile Table 6)\n")
cat("    - TC_table_best_latex.txt  (LaTeX)\n")
cat("============================================================\n")

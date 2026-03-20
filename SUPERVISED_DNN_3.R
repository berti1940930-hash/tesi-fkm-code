###########################################################################
###########################################################################
###########   TESI – SUPERVISED_DNN_GRIGNANI  (Batch top fuzzy)  ##########
###########################################################################
###########################################################################

fmt_tau_tag <- function(x) {
  gsub("\\.", "p", format(as.numeric(x), trim = TRUE, scientific = FALSE))
}

ENGINE_PATH <- "/Users/macbookpro13/Desktop/TESI copy/04_Codice/R/SUPERVISED_DNN_C7m1.1_noTau.R"
INPUT_DIR   <- "/Users/macbookpro13/Desktop/TESI copy/05_Outputs"
OUT_DIR     <- "/Users/macbookpro13/Desktop/TESI copy/05_Outputs/GRIGNANI"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

configs <- data.frame(
  C   = c(7, 7, 7, 6, 6, 7, 9, 5),
  m   = c(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1),
  tau = c(0.7, 0.8, 0.6, 0.95, 0.85, 0.85, 0.7, 0.8),
  stringsAsFactors = FALSE
)

summary_rows <- list()

cat("===================================================\n")
cat("BATCH SUPERVISED DNN – GRIGNANI\n")
cat("Engine:", ENGINE_PATH, "\n")
cat("Input dir:", INPUT_DIR, "\n")
cat("Output dir:", OUT_DIR, "\n")
cat("Configurazioni:", nrow(configs), "\n")
cat("===================================================\n\n")

for (i in seq_len(nrow(configs))) {
  C_i   <- configs$C[i]
  m_i   <- configs$m[i]
  tau_i <- configs$tau[i]

  config_key <- paste0("C", C_i, "_m", m_i)
  run_label  <- paste0(config_key, "_tau", fmt_tau_tag(tau_i), "_valTune_v7")
  perf_file  <- file.path(OUT_DIR, paste0("sup_perf_", run_label, ".csv"))
  tune_file  <- file.path(OUT_DIR, paste0("sup_tune_", run_label, ".csv"))
  perf_rds   <- file.path(OUT_DIR, paste0("sup_perf_", run_label, ".rds"))
  bt_rds     <- file.path(OUT_DIR, paste0("sup_bt_", run_label, ".rds"))
  t7_rds     <- file.path(OUT_DIR, paste0("sup_table7_", run_label, ".rds"))

  cat("[", i, "/", nrow(configs), "] ", config_key, " | tau=", tau_i, "\n", sep = "")

  Sys.setenv(
    TESI_SUP_INPUT_DIR = INPUT_DIR,
    TESI_SUP_OUT_DIR   = OUT_DIR,
    TESI_CURRENT_CONFIG = config_key,
    TESI_FIXED_TAU      = as.character(tau_i),
    TESI_RUN_LABEL      = run_label
  )

  status_i <- "ok"
  msg_i    <- ""

  if (file.exists(perf_rds) && file.exists(bt_rds) && file.exists(t7_rds)) {
    status_i <- "cache"
    msg_i    <- "Run già presente: riuso output esistenti."
    cat("  CACHE: output già presenti, salto il rerun.\n")
  } else {
    tryCatch(
      {
        source(ENGINE_PATH, local = new.env(parent = globalenv()))
      },
      error = function(e) {
        if (inherits(e, "tesi_cache_hit")) {
          status_i <<- "cache"
          msg_i <<- conditionMessage(e)
          cat("  CACHE: output già presenti, riuso i file esistenti.\n")
        } else {
          status_i <<- "error"
          msg_i <<- conditionMessage(e)
          cat("  ERROR:", msg_i, "\n")
        }
      }
    )
  }

  if (status_i %in% c("ok", "cache") && file.exists(perf_file)) {
    perf_i <- read.csv(perf_file, stringsAsFactors = FALSE)
    tune_i <- if (file.exists(tune_file)) read.csv(tune_file, stringsAsFactors = FALSE) else NULL
    best_i <- if (!is.null(tune_i) && nrow(tune_i) > 0) tune_i[1, , drop = FALSE] else NULL

    row_i <- data.frame(
      run_label      = run_label,
      config_key     = config_key,
      C              = C_i,
      m              = m_i,
      tau            = tau_i,
      status         = status_i,
      error_message  = if (status_i == "cache") msg_i else "",
      mean_ret_pct   = perf_i$mean_ret_pct[1],
      total_ret_pct  = perf_i$total_ret_pct[1],
      sharpe         = perf_i$sharpe[1],
      sortino        = perf_i$sortino[1],
      max_dd_pct     = perf_i$max_dd_pct[1],
      calmar         = perf_i$calmar[1],
      profit_factor  = perf_i$profit_factor[1],
      pct_no_trade   = perf_i$pct_no_trade[1],
      n_months       = perf_i$n_months[1],
      long_classes   = if (!is.null(best_i)) best_i$long_classes[1] else NA_character_,
      short_classes  = if (!is.null(best_i)) best_i$short_classes[1] else NA_character_,
      q_long         = if (!is.null(best_i)) best_i$q_long[1] else NA_real_,
      q_short        = if (!is.null(best_i)) best_i$q_short[1] else NA_real_,
      min_cluster    = if (!is.null(best_i)) best_i$min_cluster[1] else NA_real_,
      val_mean_ret_pct = if (!is.null(best_i)) best_i$mean_ret_pct[1] else NA_real_,
      val_sharpe       = if (!is.null(best_i)) best_i$sharpe[1] else NA_real_,
      stringsAsFactors = FALSE
    )
  } else {
    row_i <- data.frame(
      run_label      = run_label,
      config_key     = config_key,
      C              = C_i,
      m              = m_i,
      tau            = tau_i,
      status         = status_i,
      error_message  = msg_i,
      mean_ret_pct   = NA_real_,
      total_ret_pct  = NA_real_,
      sharpe         = NA_real_,
      sortino        = NA_real_,
      max_dd_pct     = NA_real_,
      calmar         = NA_real_,
      profit_factor  = NA_real_,
      pct_no_trade   = NA_real_,
      n_months       = NA_real_,
      long_classes   = NA_character_,
      short_classes  = NA_character_,
      q_long         = NA_real_,
      q_short        = NA_real_,
      min_cluster    = NA_real_,
      val_mean_ret_pct = NA_real_,
      val_sharpe       = NA_real_,
      stringsAsFactors = FALSE
    )
  }

  summary_rows[[length(summary_rows) + 1L]] <- row_i
  cat("\n")
}

Sys.unsetenv(c(
  "TESI_SUP_INPUT_DIR",
  "TESI_SUP_OUT_DIR",
  "TESI_CURRENT_CONFIG",
  "TESI_FIXED_TAU",
  "TESI_RUN_LABEL"
))

summary_df <- do.call(rbind, summary_rows)
summary_df <- summary_df[order(-summary_df$sharpe, -summary_df$mean_ret_pct), ]

summary_csv <- file.path(OUT_DIR, "grignani_supervised_summary.csv")
write.csv(summary_df, summary_csv, row.names = FALSE)

cat("===================================================\n")
cat("BATCH COMPLETATO\n")
cat("Summary:", summary_csv, "\n")
cat("===================================================\n")
print(summary_df)

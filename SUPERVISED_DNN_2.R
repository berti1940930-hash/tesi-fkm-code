###########################################################################
###########################################################################
###########   TESI – SUPERVISED_DNN_C7m1.1_noTau  (Capitolo 5)   #########
###########################################################################
###########################################################################
#
#  File autonomo per la fase supervisionata.
#  Prerequisito: UNSUPERVISED.R già eseguito (output in OUT_DIR).
#
#  Configurazione: C=7, m=1.1, tau=0 (nessun filtro di membership).
#
#  ── ARCHITETTURA (Han 2021 / Vitale 2025) ────────────────────────────────
#
#  UN SOLO DNN globale per ciascuno dei 3 feature set.
#  I cluster NON entrano nel training; vengono usati esclusivamente nella
#  fase di trading per costruire il segnale macro di direzione.
#
#  Feature set:
#    Momentum : nMOM(1,3,6,9,12) + M_MOM(1,3,6,9,12) = 10 feature
#    Return   : nRET(1..12) + M_RET(1..12)           = 24 feature
#    Hybrid   : nRET(1..12) + M_RET(1..12) + nfc_*   = 24 + n_chars feature
#
#  Target: 10 decili calcolati sull'INTERO UNIVERSO per mese
#    (ntile over all stocks per month, NOT within-cluster)
#
#  DNN (Vitale 2025 / Han 2021):
#    3 hidden layers × 64 neuroni, ReLU, Softmax(10), CCE
#    Ensemble: 5 modelli
#
#  Split temporale:
#    Training:   Feb 2004 – Dic 2015  (143 mesi etichettabili)
#    Validation: Gen 2016 – Dic 2016  ( 12 mesi)
#    Test OOS:   Gen 2017 – Nov 2020  ( 47 mesi etichettabili)
#
#  Trading rule (v8):
#    Per ogni cluster k al mese t:
#      cluster_mom = mean(mom_1) sul cluster COMPLETO (pre-tau)
#      Filtro tau SOLO per la selezione dei titoli da tradare
#      Se cluster_mom > 0 → LONG  top-q% per score p(long_classes)
#      Se cluster_mom < 0 → SHORT top-q% per score p(short_classes)
#
#  ── v8 CHANGELOG (rispetto a v7) ────────────────────────────────────────
#  FIX 7: cluster_mom calcolato sul cluster COMPLETO, prima del filtro tau.
#          In v7 cluster_mom veniva calcolato su sub_k (già tau-filtrato),
#          introducendo rumore nel segnale direzionale macro quando tau è
#          alto e il cluster viene "svuotato" dei titoli ambigui.
#          Ora: segnale direzionale = mean(mom_1) sul panel_t intero per
#          cluster k; filtro tau = solo per la selezione dei titoli.
#  FIX 8: classi long/short FISSE (non data-driven da F1/recall validation).
#          Pool di validazione in v7 costruito da val_class_metrics era
#          fragile su soli 12 mesi: rischio di ottimizzare su rumore.
#          Classi fisse plausibili dalla letteratura:
#            long_sets  = {9-10}, {10}, {9}
#            short_sets = {1-3-4}, {1-4}, {1}, {1-2}
#          Tuning su validation solo su: q_long, q_short, min_cluster.
#
###########################################################################
###########################################################################


###########################################################################
# 0. PACCHETTI
###########################################################################

library(clue)
library(fclust)
library(data.table)
library(lubridate)
library(tidyr)
library(dplyr)
library(torch)
library(parallel)

get_env_string <- function(name, default = "") {
  val <- Sys.getenv(name, unset = default)
  if (!nzchar(val)) default else val
}

get_env_numeric <- function(name, default = NA_real_) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  val <- suppressWarnings(as.numeric(raw))
  if (is.na(val)) default else val
}

fmt_tau_tag <- function(x) {
  gsub("\\.", "p", format(as.numeric(x), trim = TRUE, scientific = FALSE))
}

fmt_class_set <- function(x) {
  paste(sort(unique(as.integer(x))), collapse = "-")
}


###########################################################################
# 1. CONFIGURAZIONE
###########################################################################

N_CORES <- min(detectCores() - 2L, 8L)
cat("Core disponibili:", detectCores(), "| Core usati:", N_CORES, "\n")

INPUT_DIR <- get_env_string("TESI_SUP_INPUT_DIR", "/Users/macbookpro13/Desktop/TESI copy/05_Outputs")
OUT_DIR   <- get_env_string("TESI_SUP_OUT_DIR", INPUT_DIR)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cat("Input dir:", INPUT_DIR, "\n")
cat("Output dir:", OUT_DIR, "\n")

CURRENT_CONFIG <- get_env_string("TESI_CURRENT_CONFIG", "C7_m1.1")
CURRENT_TAU    <- get_env_numeric("TESI_FIXED_TAU", 0.0)
N_CLASSES      <- 10L
N_ENSEMBLE     <- 5L
DNN_EPOCHS     <- 500L
DNN_PATIENCE   <- 50L
DNN_LR         <- 1e-3
DNN_HIDDEN     <- 64L
SUP_LAG_FIRM_CHARS <- FALSE
SUP_LAG_FC_MONTHS  <- 1L
RUN_SUFFIX         <- paste0("_fixedClass",
                             if (SUP_LAG_FIRM_CHARS) paste0("_supLagFC", SUP_LAG_FC_MONTHS) else "")
CACHE_VERSION  <- "v8"
DEFAULT_RUN_LABEL <- paste0(
  CURRENT_CONFIG, "_tau", fmt_tau_tag(CURRENT_TAU), RUN_SUFFIX, "_", CACHE_VERSION
)
CURRENT_RUN <- get_env_string("TESI_RUN_LABEL", DEFAULT_RUN_LABEL)

TRAIN_START <- "2004-02";  TRAIN_END <- "2015-12"
VAL_START   <- "2016-01";  VAL_END   <- "2016-12"
TEST_START  <- "2017-01"

out_file_t7     <- file.path(OUT_DIR, paste0("sup_table7_", CURRENT_RUN, ".rds"))
out_file_t8     <- file.path(OUT_DIR, paste0("sup_table8_", CURRENT_RUN, ".rds"))
out_file_perf   <- file.path(OUT_DIR, paste0("sup_perf_",   CURRENT_RUN, ".rds"))
out_file_bt     <- file.path(OUT_DIR, paste0("sup_bt_",     CURRENT_RUN, ".rds"))
out_file_tune   <- file.path(OUT_DIR, paste0("sup_tune_",   CURRENT_RUN, ".csv"))

cat("===================================================\n")
cat("  Run:", CURRENT_RUN, "\n")
cat("  Config:", CURRENT_CONFIG, "| tau:", CURRENT_TAU, "\n")
cat("  FC lag test:", SUP_LAG_FIRM_CHARS,
    "| mesi:", if (SUP_LAG_FIRM_CHARS) SUP_LAG_FC_MONTHS else 0L, "\n")
cat("  3 modelli: Momentum / Return / Hybrid | Ensemble:", N_ENSEMBLE, "\n")
cat("  Trading rule: cluster_mom su cluster COMPLETO (v8)\n")
cat("  Classi: FISSE {long: 9-10/10/9} × {short: 1-3-4/1-4/1/1-2}\n")
cat("===================================================\n\n")


###########################################################################
# 2. CARICAMENTO DATI
###########################################################################

cat("--- Caricamento dati unsupervised ---\n")

refit_path <- file.path(INPUT_DIR, "refit_10configs_RS20.rds")
if (!file.exists(refit_path)) stop("refit_10configs_RS20.rds non trovato.")
refit_results <- readRDS(refit_path)
cat("  [refit_results] letto da:", refit_path, "\n")

if (!CURRENT_CONFIG %in% names(refit_results))
  stop("Config '", CURRENT_CONFIG, "' non trovata. Disponibili: ",
       paste(names(refit_results), collapse=", "))

final_df_path <- file.path(INPUT_DIR, "final_df.rds")
if (!file.exists(final_df_path)) stop("final_df.rds non trovato.")
final_df <- readRDS(final_df_path)
if (!"ym" %in% names(final_df))
  final_df <- final_df %>% mutate(ym = format(date, "%Y-%m"))
final_df$ym     <- as.character(final_df$ym)
final_df$permno <- as.character(final_df$permno)

if (SUP_LAG_FIRM_CHARS) {
  fc_lag_excl <- c("permno", "ym", "date", "ret", "price", "log1pr",
                   paste0("mom_", 1:48))
  fc_lag_cols <- setdiff(names(final_df), fc_lag_excl)
  fc_lag_cols <- fc_lag_cols[sapply(final_df[, fc_lag_cols, drop=FALSE], is.numeric)]
  n_before <- nrow(final_df)
  final_df <- final_df %>%
    arrange(permno, date) %>%
    group_by(permno) %>%
    mutate(across(all_of(fc_lag_cols), ~ dplyr::lag(.x, n = SUP_LAG_FC_MONTHS))) %>%
    ungroup() %>%
    filter(complete.cases(across(all_of(fc_lag_cols))))
  cat("  [FC LAG] Applicato lag di", SUP_LAG_FC_MONTHS, "mese/i a",
      length(fc_lag_cols), "firm characteristics.\n")
  cat("  [FC LAG] Righe:", n_before, "->", nrow(final_df),
      "| mesi:", length(unique(final_df$ym)), "\n")
}

months_sorted <- sort(unique(final_df$ym))
cat("  [final_df] letto da:", final_df_path, "\n")
cat("  Righe:", nrow(final_df),
    "| Permno:", length(unique(final_df$permno)),
    "| Mesi:", length(months_sorted),
    "| Da:", months_sorted[1], "a:", months_sorted[length(months_sorted)], "\n\n")


###########################################################################
# 3. IDENTIFICAZIONE FIRM CHARACTERISTICS
###########################################################################

mom_cols  <- paste0("mom_", 1:48)
excl_cols <- c("permno", "ym", "date", "ret", "price", "log1pr",
               "ret_next", mom_cols)

char_cols <- setdiff(names(final_df), excl_cols)
char_cols <- char_cols[sapply(final_df[, char_cols, drop=FALSE], is.numeric)]
# NESSUN regex: vogliamo tutte le firm characteristics numeriche

if (length(char_cols) == 0) stop("Nessuna firm characteristic trovata.")
cat("Firm characteristics:", length(char_cols), "\n")
cat("  Prime 5:", paste(head(char_cols, 5), collapse=", "), "\n\n")


###########################################################################
# 4. COSTRUZIONE FEATURE SET
###########################################################################

cat("Costruzione feature set...\n")

final_df <- final_df %>%
  arrange(permno, ym) %>%
  group_by(permno) %>%
  mutate(ret_next = lead(ret)) %>%
  ungroup()

feat_df <- final_df %>%
  mutate(
    RET_1  = mom_1,
    RET_2  = mom_2,
    RET_3  = (1 + mom_3)  / (1 + mom_2)  - 1,
    RET_4  = (1 + mom_4)  / (1 + mom_3)  - 1,
    RET_5  = (1 + mom_5)  / (1 + mom_4)  - 1,
    RET_6  = (1 + mom_6)  / (1 + mom_5)  - 1,
    RET_7  = (1 + mom_7)  / (1 + mom_6)  - 1,
    RET_8  = (1 + mom_8)  / (1 + mom_7)  - 1,
    RET_9  = (1 + mom_9)  / (1 + mom_8)  - 1,
    RET_10 = (1 + mom_10) / (1 + mom_9)  - 1,
    RET_11 = (1 + mom_11) / (1 + mom_10) - 1,
    RET_12 = (1 + mom_12) / (1 + mom_11) - 1
  )

feat_df <- feat_df %>%
  group_by(ym) %>%
  mutate(
    M_MOM1  = mean(mom_1,  na.rm=TRUE), S_MOM1  = sd(mom_1,  na.rm=TRUE),
    M_MOM3  = mean(mom_3,  na.rm=TRUE), S_MOM3  = sd(mom_3,  na.rm=TRUE),
    M_MOM6  = mean(mom_6,  na.rm=TRUE), S_MOM6  = sd(mom_6,  na.rm=TRUE),
    M_MOM9  = mean(mom_9,  na.rm=TRUE), S_MOM9  = sd(mom_9,  na.rm=TRUE),
    M_MOM12 = mean(mom_12, na.rm=TRUE), S_MOM12 = sd(mom_12, na.rm=TRUE),
    nMOM1   = (mom_1  - M_MOM1)  / pmax(S_MOM1,  1e-8),
    nMOM3   = (mom_3  - M_MOM3)  / pmax(S_MOM3,  1e-8),
    nMOM6   = (mom_6  - M_MOM6)  / pmax(S_MOM6,  1e-8),
    nMOM9   = (mom_9  - M_MOM9)  / pmax(S_MOM9,  1e-8),
    nMOM12  = (mom_12 - M_MOM12) / pmax(S_MOM12, 1e-8)
  ) %>%
  ungroup()

feat_df <- feat_df %>%
  group_by(ym) %>%
  mutate(
    M_RET1  = mean(RET_1,  na.rm=TRUE), S_RET1  = sd(RET_1,  na.rm=TRUE),
    M_RET2  = mean(RET_2,  na.rm=TRUE), S_RET2  = sd(RET_2,  na.rm=TRUE),
    M_RET3  = mean(RET_3,  na.rm=TRUE), S_RET3  = sd(RET_3,  na.rm=TRUE),
    M_RET4  = mean(RET_4,  na.rm=TRUE), S_RET4  = sd(RET_4,  na.rm=TRUE),
    M_RET5  = mean(RET_5,  na.rm=TRUE), S_RET5  = sd(RET_5,  na.rm=TRUE),
    M_RET6  = mean(RET_6,  na.rm=TRUE), S_RET6  = sd(RET_6,  na.rm=TRUE),
    M_RET7  = mean(RET_7,  na.rm=TRUE), S_RET7  = sd(RET_7,  na.rm=TRUE),
    M_RET8  = mean(RET_8,  na.rm=TRUE), S_RET8  = sd(RET_8,  na.rm=TRUE),
    M_RET9  = mean(RET_9,  na.rm=TRUE), S_RET9  = sd(RET_9,  na.rm=TRUE),
    M_RET10 = mean(RET_10, na.rm=TRUE), S_RET10 = sd(RET_10, na.rm=TRUE),
    M_RET11 = mean(RET_11, na.rm=TRUE), S_RET11 = sd(RET_11, na.rm=TRUE),
    M_RET12 = mean(RET_12, na.rm=TRUE), S_RET12 = sd(RET_12, na.rm=TRUE),
    nRET1   = (RET_1  - M_RET1)  / pmax(S_RET1,  1e-8),
    nRET2   = (RET_2  - M_RET2)  / pmax(S_RET2,  1e-8),
    nRET3   = (RET_3  - M_RET3)  / pmax(S_RET3,  1e-8),
    nRET4   = (RET_4  - M_RET4)  / pmax(S_RET4,  1e-8),
    nRET5   = (RET_5  - M_RET5)  / pmax(S_RET5,  1e-8),
    nRET6   = (RET_6  - M_RET6)  / pmax(S_RET6,  1e-8),
    nRET7   = (RET_7  - M_RET7)  / pmax(S_RET7,  1e-8),
    nRET8   = (RET_8  - M_RET8)  / pmax(S_RET8,  1e-8),
    nRET9   = (RET_9  - M_RET9)  / pmax(S_RET9,  1e-8),
    nRET10  = (RET_10 - M_RET10) / pmax(S_RET10, 1e-8),
    nRET11  = (RET_11 - M_RET11) / pmax(S_RET11, 1e-8),
    nRET12  = (RET_12 - M_RET12) / pmax(S_RET12, 1e-8)
  ) %>%
  ungroup()

feat_df <- feat_df %>%
  group_by(ym) %>%
  mutate(across(
    all_of(char_cols),
    ~ (. - mean(., na.rm=TRUE)) / pmax(sd(., na.rm=TRUE), 1e-8),
    .names = "nfc_{.col}"
  )) %>%
  ungroup()

nmom_cols <- c("nMOM1", "nMOM3", "nMOM6", "nMOM9", "nMOM12")
mmom_cols <- c("M_MOM1", "M_MOM3", "M_MOM6", "M_MOM9", "M_MOM12")
nret_cols <- paste0("nRET",  1:12)
mret_cols <- paste0("M_RET", 1:12)
nfc_cols  <- paste0("nfc_",  char_cols)

feat_momentum <- c(nmom_cols, mmom_cols)
feat_return   <- c(nret_cols, mret_cols)
feat_hybrid   <- c(nret_cols, mret_cols, nfc_cols)

cat("Feature set:\n")
cat("  Momentum : nMOM + M_MOM                  =", length(feat_momentum), "\n")
cat("  Return   : nRET(1..12) + M_RET(1..12)    =", length(feat_return),   "\n")
cat("  Hybrid   : Return + nfc_*                 =", length(feat_hybrid),   "\n\n")

for (fc in list(feat_momentum, feat_return, feat_hybrid)) {
  miss <- setdiff(fc, names(feat_df))
  if (length(miss) > 0) stop("Feature mancanti: ", paste(head(miss,5), collapse=", "))
}

feat_df$permno <- as.character(feat_df$permno)
feat_df$ym     <- as.character(feat_df$ym)


###########################################################################
# 5. CLUSTER ASSIGNMENT + ALLINEAMENTO HUNGARIAN
###########################################################################

cat("Estrazione + allineamento cluster per", CURRENT_CONFIG, "...\n")

config_fits <- refit_results[[CURRENT_CONFIG]]
K_sel       <- as.integer(sub("C(\\d+)_.*", "\\1", CURRENT_CONFIG))

compute_soft_centroids <- function(U, X) {
  X[!is.finite(X)] <- 0
  K <- ncol(U)
  V <- matrix(0.0, K, ncol(X))
  for (k in seq_len(K)) {
    w  <- U[, k]; w[!is.finite(w)] <- 0; sw <- sum(w)
    V[k, ] <- if (sw > 1e-12) colSums(w * X) / sw else rep(0.0, ncol(X))
  }
  V
}

align_hungarian <- function(V_ref, V_new) {
  p_use <- seq_len(min(ncol(V_ref), ncol(V_new)))
  K <- nrow(V_ref)
  D <- matrix(0.0, K, K)
  for (i in seq_len(K)) for (j in seq_len(K)) {
    d <- sum((V_new[i, p_use] - V_ref[j, p_use])^2)
    D[i, j] <- if (is.finite(d)) d else .Machine$double.xmax
  }
  D[D < 0] <- 0
  as.integer(clue::solve_LSAP(D))
}

has_clue <- requireNamespace("clue", quietly=TRUE)
pca_path <- file.path(INPUT_DIR, "pca_panel_99pct.rds")
has_pca  <- file.exists(pca_path)
do_align <- has_clue && has_pca

if (!has_clue) cat("  [WARN] 'clue' non trovato — uso etichette raw.\n")
if (!has_pca)  cat("  [WARN] 'pca_panel_99pct.rds' non trovato — uso etichette raw.\n")

if (do_align) {
  cat("  Carico pca_panel_99pct.rds...\n")
  pca_sup <- readRDS(pca_path)
  pca_sup$permno <- as.character(pca_sup$permno)
  pca_sup$ym     <- as.character(pca_sup$ym)
}

months_fit    <- sort(names(config_fits))
aligned_list  <- vector("list", length(months_fit))
names(aligned_list) <- months_fit
V_ref     <- NULL
n_aligned <- 0L

for (i in seq_along(months_fit)) {
  mm     <- months_fit[i]
  fit_mm <- config_fits[[mm]]
  if (is.null(fit_mm) || is.null(fit_mm$permno) ||
      is.null(fit_mm$assign) || is.null(fit_mm$U)) next

  permnos_mm <- as.character(fit_mm$permno)
  assign_mm  <- as.integer(fit_mm$assign)
  max_u_mm   <- apply(fit_mm$U, 1, max)

  if (do_align) {
    pca_mm     <- pca_sup[pca_sup$ym == mm, , drop=FALSE]
    pc_cols_mm <- grep("^PC[0-9]+$", names(pca_mm), value=TRUE)
    pc_cols_mm <- pc_cols_mm[order(as.integer(sub("^PC", "", pc_cols_mm)))]
    idx        <- match(permnos_mm, pca_mm$permno)
    valid      <- !is.na(idx)

    if (sum(valid) >= K_sel && length(pc_cols_mm) >= 1) {
      X_mm <- as.matrix(pca_mm[idx[valid], pc_cols_mm, drop=FALSE])
      U_mm <- fit_mm$U[valid, , drop=FALSE]
      V_mm <- compute_soft_centroids(U_mm, X_mm)

      if (is.null(V_ref)) {
        V_ref <- V_mm
      } else {
        perm      <- align_hungarian(V_ref, V_mm)
        assign_mm <- perm[assign_mm]
        inv_perm  <- order(perm)
        V_ref     <- V_mm[inv_perm, , drop=FALSE]
        n_aligned <- n_aligned + 1L
      }
    }
  }

  aligned_list[[mm]] <- data.frame(
    ym=mm, permno=permnos_mm, cluster=assign_mm,
    max_u=max_u_mm, stringsAsFactors=FALSE
  )
}

assign_df    <- do.call(rbind, Filter(Negate(is.null), aligned_list))
assign_df$ym <- as.character(assign_df$ym)
align_status <- if (do_align) paste0("HUNGARIAN (", n_aligned, " mesi allineati)") else "RAW"
cat("  assign_df:", nrow(assign_df), "righe | mesi:", length(unique(assign_df$ym)),
    "| cluster:", length(unique(assign_df$cluster)),
    "\n  Allineamento:", align_status, "\n\n")
if (nrow(assign_df) == 0) stop("assign_df vuoto.")


###########################################################################
# 6. COSTRUZIONE model_df — DECILI GLOBALI
###########################################################################

cat("Costruzione model_df (decili globali)...\n")

cols_base  <- unique(c(
  "permno", "ym", "ret_next", "mom_1",
  feat_momentum,
  feat_return,
  feat_hybrid
))
cols_avail <- intersect(cols_base, names(feat_df))

model_df <- feat_df %>%
  select(all_of(cols_avail)) %>%
  filter(!is.na(ret_next)) %>%
  group_by(ym) %>%
  mutate(decile_label = ntile(ret_next, 10L)) %>%
  ungroup() %>%
  filter(!is.na(decile_label))

cat("  model_df:", nrow(model_df), "righe",
    "| mesi:", length(unique(model_df$ym)),
    "| decili distinti:", length(unique(model_df$decile_label)), "\n\n")
if (nrow(model_df) == 0) stop("model_df vuoto.")


###########################################################################
# 7. SPLIT TEMPORALE
###########################################################################

df_tr <- model_df %>% filter(ym >= TRAIN_START & ym <= TRAIN_END)
df_va <- model_df %>% filter(ym >= VAL_START   & ym <= VAL_END)
df_te <- model_df %>% filter(ym >= TEST_START)

cat("Split temporale:\n")
cat(sprintf("  Training:   %5d obs | %d mesi\n", nrow(df_tr), length(unique(df_tr$ym))))
cat(sprintf("  Validation: %5d obs | %d mesi\n", nrow(df_va), length(unique(df_va$ym))))
cat(sprintf("  Test OOS:   %5d obs | %d mesi\n", nrow(df_te), length(unique(df_te$ym))))
cat("\n")
if (nrow(df_tr) == 0) stop("Training set vuoto.")
if (nrow(df_va) == 0) stop("Validation set vuoto.")


###########################################################################
# 8. SETUP TORCH
###########################################################################

torch_set_num_threads(N_CORES)
DNN_DEVICE <- tryCatch({
  if (isTRUE(cuda_is_available()))           { cat("Device: CUDA\n");       torch_device("cuda") }
  else if (isTRUE(torch_is_mps_available())) { cat("Device: MPS\n");        torch_device("mps")  }
  else                                       { cat("Device: CPU\n");         torch_device("cpu")  }
}, error = function(e) { cat("Device: CPU\n"); torch_device("cpu") })

ClusterDNN <- nn_module(
  "ClusterDNN",
  initialize = function(n_in, n_out = 10L, hidden = 64L) {
    self$fc1 <- nn_linear(n_in, hidden)
    self$fc2 <- nn_linear(hidden, hidden)
    self$fc3 <- nn_linear(hidden, hidden)
    self$out  <- nn_linear(hidden, n_out)
  },
  forward = function(x) {
    x <- nnf_relu(self$fc1(x))
    x <- nnf_relu(self$fc2(x))
    x <- nnf_relu(self$fc3(x))
    self$out(x)
  }
)


###########################################################################
# 9. FUNZIONI DI TRAINING
###########################################################################

train_one_dnn <- function(X_tr_t, y_tr_t, X_va_t, y_va_t, n_in,
                           epochs=DNN_EPOCHS, lr=DNN_LR,
                           patience=DNN_PATIENCE, batch_sz=256L) {
  X_tr_t <- X_tr_t$to(device=DNN_DEVICE); y_tr_t <- y_tr_t$to(device=DNN_DEVICE)
  X_va_t <- X_va_t$to(device=DNN_DEVICE); y_va_t <- y_va_t$to(device=DNN_DEVICE)

  model   <- ClusterDNN(n_in=n_in, n_out=N_CLASSES, hidden=DNN_HIDDEN)$to(device=DNN_DEVICE)
  opt     <- optim_adam(model$parameters, lr=lr, weight_decay=1e-5)
  loss_fn <- nn_cross_entropy_loss()
  sched   <- lr_reduce_on_plateau(opt, factor=0.5, patience=15L, min_lr=1e-5)

  best_loss <- Inf; best_state <- NULL; wait <- 0L
  n_tr <- y_tr_t$shape[1]

  for (ep in seq_len(epochs)) {
    model$train()
    idx <- sample(n_tr)
    for (s in seq(1L, n_tr, by=batch_sz)) {
      b <- idx[s:min(s+batch_sz-1L, n_tr)]
      opt$zero_grad()
      loss_fn(model(X_tr_t[b, ]), y_tr_t[b])$backward()
      opt$step()
    }
    model$eval()
    with_no_grad({ val_loss <- loss_fn(model(X_va_t), y_va_t)$item() })
    sched$step(val_loss)
    if (val_loss < best_loss - 1e-6) {
      best_loss  <- val_loss
      best_state <- lapply(model$state_dict(), function(t) t$clone())
      wait <- 0L
    } else {
      wait <- wait + 1L
      if (wait >= patience) break
    }
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  model$eval(); model
}

sanitize_matrix <- function(X_raw) {
  X_mat <- as.matrix(X_raw)
  X_mat[!is.finite(X_mat)] <- 0
  X_mat
}

train_global_ensemble <- function(feat_cols_sel, df_train, df_val,
                                   n_ens=N_ENSEMBLE) {
  X_tr_s <- sanitize_matrix(df_train[, feat_cols_sel])
  X_va_s <- sanitize_matrix(df_val[,   feat_cols_sel])

  n_in <- length(feat_cols_sel)
  cat(sprintf("    %d obs × %d feature\n", nrow(X_tr_s), n_in))

  X_tr_t <- torch_tensor(X_tr_s, dtype=torch_float())
  y_tr_t <- torch_tensor(as.integer(df_train$decile_label), dtype=torch_long())
  X_va_t <- torch_tensor(X_va_s, dtype=torch_float())
  y_va_t <- torch_tensor(as.integer(df_val$decile_label),   dtype=torch_long())

  ensemble <- vector("list", n_ens)
  for (i in seq_len(n_ens)) {
    cat("    Membro", i, "/", n_ens, "...")
    ensemble[[i]] <- train_one_dnn(X_tr_t, y_tr_t, X_va_t, y_va_t, n_in=n_in)
    cat(" ok\n")
  }
  list(ensemble=ensemble, feat_cols=feat_cols_sel)
}

predict_probs <- function(model_obj, X_raw) {
  X_s <- sanitize_matrix(X_raw)
  Xt  <- torch_tensor(X_s, dtype=torch_float())$to(device=DNN_DEVICE)
  prob_sum <- NULL
  for (m in model_obj$ensemble) {
    m$eval()
    p <- with_no_grad({ as.matrix(nnf_softmax(m(Xt), dim=2)$cpu()) })
    if (is.null(prob_sum)) prob_sum <- p else prob_sum <- prob_sum + p
  }
  prob_sum / length(model_obj$ensemble)
}


###########################################################################
# 10. CHECK CACHE
###########################################################################

if (file.exists(out_file_perf) && file.exists(out_file_bt) &&
    file.exists(out_file_t7)) {
  cat("[CACHE] Run", CURRENT_RUN, "già presente — carico da disco\n")
  perf_df    <- readRDS(out_file_perf)
  bt_df      <- readRDS(out_file_bt)
  dnn_table7 <- readRDS(out_file_t7)
  cat("\n--- Table 7 ---\n"); print(dnn_table7)
  cat("\n--- Performance ---\n"); print(perf_df)
  cat("\nFine (cache).\n")
  stop(structure(
    list(message = paste0("CACHE_HIT:", CURRENT_RUN), call = NULL),
    class = c("tesi_cache_hit", "error", "condition")
  ))
}


###########################################################################
# 11. TRAINING — TRE MODELLI GLOBALI
###########################################################################

torch_manual_seed(42L); set.seed(42L)
t_start <- Sys.time()
cat("Avvio training:", format(Sys.time(), "%H:%M:%S"), "\n\n")

model_configs <- list(
  Momentum = feat_momentum,
  Return   = feat_return,
  Hybrid   = feat_hybrid
)

trained_models <- list()

for (mname in names(model_configs)) {
  fc <- model_configs[[mname]]
  cat(">>> Modello:", mname, "| Feature:", length(fc), "\n")
  trained_models[[mname]] <- train_global_ensemble(fc, df_tr, df_va)
  cat("\n")
}

cat("Training completato in",
    round(difftime(Sys.time(), t_start, units="mins"), 1), "min\n\n")


###########################################################################
# 12. FUNZIONI DI VALUTAZIONE
###########################################################################

eval_split_metrics <- function(model_obj, df_split, split_name, model_name) {
  if (is.null(df_split) || nrow(df_split) == 0) return(NULL)
  fc   <- model_obj$feat_cols
  X    <- as.matrix(df_split[, fc])
  y    <- as.integer(df_split$decile_label)
  loss_fn <- nn_cross_entropy_loss()
  X_s  <- sanitize_matrix(X)
  Xt   <- torch_tensor(X_s, dtype=torch_float())$to(device=DNN_DEVICE)
  yt   <- torch_tensor(y,   dtype=torch_long())$to(device=DNN_DEVICE)
  loss <- mean(sapply(model_obj$ensemble, function(m) {
    m$eval(); with_no_grad(loss_fn(m(Xt), yt)$item())
  }))
  probs  <- predict_probs(model_obj, X)
  y_pred <- apply(probs, 1, which.max)
  data.frame(run=CURRENT_RUN, feature_set=model_name, split=split_name,
             loss=round(loss,4), accuracy=round(mean(y_pred==y),4),
             n_obs=nrow(df_split))
}

eval_class_metrics <- function(model_obj, df_split, model_name) {
  if (is.null(df_split) || nrow(df_split) == 0) return(NULL)
  fc     <- model_obj$feat_cols
  X      <- as.matrix(df_split[, fc])
  y_true <- as.integer(df_split$decile_label)
  y_pred <- apply(predict_probs(model_obj, X), 1, which.max)
  pr_rows <- lapply(seq_len(N_CLASSES), function(cl) {
    tp <- sum(y_pred==cl & y_true==cl)
    fp <- sum(y_pred==cl & y_true!=cl)
    fn <- sum(y_pred!=cl & y_true==cl)
    prec <- if ((tp+fp)>0) tp/(tp+fp) else NA_real_
    rec  <- if ((tp+fn)>0) tp/(tp+fn) else NA_real_
    f1   <- if (!is.na(prec)&&!is.na(rec)&&(prec+rec)>0) 2*prec*rec/(prec+rec) else NA_real_
    data.frame(run=CURRENT_RUN, feature_set=model_name,
               decile=cl, precision=prec, recall=rec, f1=f1)
  })
  do.call(rbind, pr_rows)
}

calc_performance <- function(bt_df, label="") {
  rets <- bt_df$ret[!is.na(bt_df$ret)]
  if (length(rets) == 0) return(NULL)
  mean_ret_y  <- mean(rets) * 12
  total_ret   <- prod(1 + rets) - 1
  sd_ann      <- sd(rets) * sqrt(12)
  sharpe      <- if (sd_ann == 0) NA_real_ else mean_ret_y / sd_ann
  down_dev_y  <- sqrt(mean(pmin(rets, 0)^2)) * sqrt(12)
  sortino     <- if (is.na(down_dev_y)||down_dev_y==0) NA_real_ else mean_ret_y / down_dev_y
  cum         <- cumprod(1 + rets)
  max_dd      <- min((cum - cummax(cum)) / cummax(cum))
  calmar      <- if (max_dd == 0) NA_real_ else mean_ret_y / abs(max_dd)
  gp <- sum(rets[rets > 0]); gl <- abs(sum(rets[rets < 0]))
  pf <- if (gl == 0) NA_real_ else gp / gl
  data.frame(config=label,
             mean_ret_pct=round(mean_ret_y*100,4), total_ret_pct=round(total_ret*100,2),
             sharpe=round(sharpe,4), sortino=round(sortino,4),
             max_dd_pct=round(max_dd*100,2), calmar=round(calmar,4),
             profit_factor=round(pf,4), pct_no_trade=round(mean(rets==0)*100,1),
             n_months=length(rets))
}


###########################################################################
# 13. TABLE 7: LOSS / ACCURACY
###########################################################################

cat("Calcolo Table 7...\n")
t7_rows <- list()
for (mname in names(trained_models)) {
  mo <- trained_models[[mname]]
  t7_rows <- c(t7_rows, list(
    eval_split_metrics(mo, df_tr, "train", mname),
    eval_split_metrics(mo, df_va, "val",   mname),
    eval_split_metrics(mo, df_te, "test",  mname)
  ))
}
dnn_table7 <- do.call(rbind, Filter(Negate(is.null), t7_rows))
cat("\n--- Table 7 ---\n")
print(dnn_table7)


###########################################################################
# 14. TABLE 8: P/R/F1 (modello Hybrid, test)
###########################################################################

cat("\nCalcolo Table 8 (Hybrid, test)...\n")
dnn_table8 <- eval_class_metrics(trained_models[["Hybrid"]], df_te, "Hybrid")
if (!is.null(dnn_table8)) { cat("\n--- Table 8 ---\n"); print(dnn_table8) }


###########################################################################
# 15. SCORING + TUNING + BACKTEST OOS (modello Hybrid)
###########################################################################

cat("\nPreparazione scoring mensile (modello Hybrid)...\n")
hybrid_model <- trained_models[["Hybrid"]]
feat_bt_cols <- intersect(c("permno", feat_hybrid, "mom_1"), names(feat_df))
months_scored <- months_sorted[months_sorted >= VAL_START]
months_scored <- months_scored[-length(months_scored)]

# ── build_scored_panels ───────────────────────────────────────────────────
# Costruisce, per ogni mese, il panel completo con probabilità DNN.
# Il panel NON applica il filtro tau — questo viene fatto in run_hybrid_bt_from_panels.
# In questo modo cluster_mom può essere calcolato sul cluster intero (FIX v8).
build_scored_panels <- function(model_obj, month_vec, feat_cols_bt,
                                feat_frame, assign_frame, ret_frame) {
  out <- list()
  for (t_month in month_vec) {
    t_idx    <- which(months_sorted == t_month)
    if (length(t_idx) == 0 || t_idx >= length(months_sorted)) next
    t1_month <- months_sorted[t_idx + 1L]

    assign_t <- assign_frame %>%
      filter(ym == t_month) %>%
      select(permno, cluster, max_u)
    if (nrow(assign_t) == 0) next

    feat_t <- feat_frame %>%
      filter(ym == t_month) %>%
      select(all_of(feat_cols_bt))
    if (nrow(feat_t) == 0) next

    rets_t1 <- ret_frame %>%
      filter(ym == t1_month) %>%
      select(permno, ret)

    panel_t <- assign_t %>%
      inner_join(feat_t, by = "permno") %>%
      inner_join(rets_t1, by = "permno") %>%
      filter(!is.na(ret))
    if (nrow(panel_t) < 4L) next

    X_all <- as.matrix(panel_t[, feat_hybrid])
    probs_all <- tryCatch(predict_probs(model_obj, X_all), error = function(e) NULL)
    if (is.null(probs_all) || ncol(probs_all) < N_CLASSES) next

    prob_df <- as.data.frame(probs_all)
    names(prob_df) <- paste0("p", seq_len(N_CLASSES))
    panel_t <- bind_cols(panel_t, prob_df)
    out[[t_month]] <- panel_t
  }
  out
}

scored_panels <- build_scored_panels(
  model_obj    = hybrid_model,
  month_vec    = months_scored,
  feat_cols_bt = feat_bt_cols,
  feat_frame   = feat_df,
  assign_frame = assign_df,
  ret_frame    = final_df
)
cat("  Mesi con score pronti:", length(scored_panels), "\n")

# ── run_hybrid_bt_from_panels ─────────────────────────────────────────────
# FIX v8: cluster_mom calcolato sul panel COMPLETO (prima del filtro tau).
# Il filtro tau viene applicato SOLO alla selezione dei titoli da tradare.
# Questo garantisce che il segnale direzionale macro rispecchi l'intero
# cluster e non sia distorto dal numero di titoli che passano il filtro.
run_hybrid_bt_from_panels <- function(scored_panels, months_use,
                                      long_classes, short_classes,
                                      tau = 0.0, q_long = 0.80, q_short = 0.80,
                                      min_cluster_size = 15L, min_sel = 3L) {
  p_long_cols  <- paste0("p", sort(unique(as.integer(long_classes))))
  p_short_cols <- paste0("p", sort(unique(as.integer(short_classes))))

  bt_rows <- lapply(months_use, function(t_month) {
    panel_t <- scored_panels[[t_month]]
    if (is.null(panel_t) || nrow(panel_t) == 0) {
      return(data.frame(ym=t_month, ret=0, n_long_cl=0L, n_short_cl=0L,
                        mean_ret_long=NA_real_, mean_ret_short=NA_real_))
    }

    # FIX v8: cluster_mom calcolato sul panel COMPLETO (senza filtro tau)
    cluster_moms <- panel_t %>%
      group_by(cluster) %>%
      summarise(cluster_mom = mean(mom_1, na.rm = TRUE), .groups = "drop")

    # Filtro tau SOLO per la selezione dei titoli
    stocks_t <- panel_t %>% filter(max_u >= tau)
    if (nrow(stocks_t) < 4L) {
      return(data.frame(ym=t_month, ret=0, n_long_cl=0L, n_short_cl=0L,
                        mean_ret_long=NA_real_, mean_ret_short=NA_real_))
    }

    cluster_rets <- numeric(0)
    n_long_cl  <- 0L; r_long_acc  <- numeric(0)
    n_short_cl <- 0L; r_short_acc <- numeric(0)

    for (k in sort(unique(stocks_t$cluster))) {
      sub_k <- stocks_t %>% filter(cluster == k)
      if (nrow(sub_k) < min_cluster_size) next

      # FIX v8: usa cluster_mom del cluster COMPLETO
      cm_val <- cluster_moms$cluster_mom[cluster_moms$cluster == k]
      cluster_mom <- if (length(cm_val) > 0) cm_val[1] else NA_real_
      if (!is.finite(cluster_mom)) next

      if (cluster_mom > 0 && length(p_long_cols) > 0) {
        score_long <- rowSums(as.matrix(sub_k[, p_long_cols, drop = FALSE]))
        th  <- quantile(score_long, q_long, na.rm = TRUE)
        sel <- score_long >= th
        if (sum(sel) < min_sel) next
        r <- mean(sub_k$ret[sel], na.rm = TRUE)
        cluster_rets <- c(cluster_rets, r)
        n_long_cl <- n_long_cl + 1L
        r_long_acc <- c(r_long_acc, r)
      } else if (cluster_mom < 0 && length(p_short_cols) > 0) {
        score_short <- rowSums(as.matrix(sub_k[, p_short_cols, drop = FALSE]))
        th  <- quantile(score_short, q_short, na.rm = TRUE)
        sel <- score_short >= th
        if (sum(sel) < min_sel) next
        r <- mean(sub_k$ret[sel], na.rm = TRUE)
        cluster_rets <- c(cluster_rets, -r)
        n_short_cl <- n_short_cl + 1L
        r_short_acc <- c(r_short_acc, r)
      }
    }

    data.frame(
      ym = t_month,
      ret = if (length(cluster_rets) == 0) 0 else mean(cluster_rets),
      n_long_cl = n_long_cl, n_short_cl = n_short_cl,
      mean_ret_long  = if (n_long_cl  == 0) NA_real_ else mean(r_long_acc),
      mean_ret_short = if (n_short_cl == 0) NA_real_ else mean(r_short_acc)
    )
  })

  do.call(rbind, bt_rows)
}

# ── Tuning su validation ──────────────────────────────────────────────────
# FIX v8: classi long/short FISSE (non data-driven da F1/recall validation).
# Il pool è stabilito a priori dalla letteratura:
#   long  = top decili (9-10, 10, 9)
#   short = bottom decili (1-3-4, 1-4, 1, 1-2)
# Tuning SOLO su: q_long, q_short, min_cluster.

# Classi fisse
long_sets  <- list(c(9L, 10L), c(10L), c(9L))
short_sets <- list(c(1L, 3L, 4L), c(1L, 4L), c(1L), c(1L, 2L))

tau_grid         <- sort(unique(c(CURRENT_TAU)))
q_long_grid      <- c(0.80, 0.85, 0.90)
q_short_grid     <- c(0.85, 0.90, 0.95)
min_cluster_grid <- c(10L, 15L)

val_months_bt  <- names(scored_panels)
val_months_bt  <- val_months_bt[val_months_bt >= VAL_START & val_months_bt < VAL_END]
test_months_bt <- names(scored_panels)
test_months_bt <- test_months_bt[test_months_bt >= TEST_START]

cat("\nTuning su validation set (classi fisse)...\n")
cat("  Long  sets :", paste(sapply(long_sets,  fmt_class_set), collapse = " | "), "\n")
cat("  Short sets :", paste(sapply(short_sets, fmt_class_set), collapse = " | "), "\n")
cat("  Tau fisso  :", paste(tau_grid, collapse = ","), "\n")
cat("  Mesi val.  :", length(val_months_bt), "\n")

tune_rows <- list()
for (long_set in long_sets) {
  for (short_set in short_sets) {
    for (tau_i in tau_grid) {
      for (q_long_i in q_long_grid) {
        for (q_short_i in q_short_grid) {
          for (min_cl_i in min_cluster_grid) {
            bt_val_i <- run_hybrid_bt_from_panels(
              scored_panels     = scored_panels,
              months_use        = val_months_bt,
              long_classes      = long_set,
              short_classes     = short_set,
              tau               = tau_i,
              q_long            = q_long_i,
              q_short           = q_short_i,
              min_cluster_size  = min_cl_i,
              min_sel           = 3L
            )
            perf_i <- calc_performance(bt_val_i, label = "val")
            if (is.null(perf_i)) next
            tune_rows[[length(tune_rows) + 1L]] <- data.frame(
              long_classes   = fmt_class_set(long_set),
              short_classes  = fmt_class_set(short_set),
              tau            = tau_i,
              q_long         = q_long_i,
              q_short        = q_short_i,
              min_cluster    = min_cl_i,
              mean_ret_pct   = perf_i$mean_ret_pct,
              total_ret_pct  = perf_i$total_ret_pct,
              sharpe         = perf_i$sharpe,
              max_dd_pct     = perf_i$max_dd_pct,
              profit_factor  = perf_i$profit_factor,
              pct_no_trade   = perf_i$pct_no_trade,
              active_months  = sum(bt_val_i$ret != 0, na.rm = TRUE),
              mean_long_pct  = round(mean(bt_val_i$mean_ret_long,  na.rm = TRUE) * 100, 4),
              mean_short_raw_pct = round(mean(bt_val_i$mean_ret_short, na.rm = TRUE) * 100, 4)
            )
          }
        }
      }
    }
  }
}

tune_df <- do.call(rbind, tune_rows)
if (is.null(tune_df) || nrow(tune_df) == 0) stop("Tuning fallito: nessuna configurazione valida.")
tune_df <- tune_df %>%
  arrange(desc(sharpe), desc(mean_ret_pct), mean_short_raw_pct, pct_no_trade)

best_cfg <- tune_df[1, , drop = FALSE]
write.csv(tune_df, out_file_tune, row.names = FALSE)

cat("\n--- Best Validation Config ---\n")
print(best_cfg)
cat("  Tuning grid salvata in:", basename(out_file_tune), "\n")

cat("\nBacktest OOS (modello Hybrid, config tuned)...\n")
bt_df <- run_hybrid_bt_from_panels(
  scored_panels     = scored_panels,
  months_use        = test_months_bt,
  long_classes      = as.integer(strsplit(best_cfg$long_classes, "-", fixed = TRUE)[[1]]),
  short_classes     = as.integer(strsplit(best_cfg$short_classes, "-", fixed = TRUE)[[1]]),
  tau               = best_cfg$tau,
  q_long            = best_cfg$q_long,
  q_short           = best_cfg$q_short,
  min_cluster_size  = best_cfg$min_cluster,
  min_sel           = 3L
)

n_traded <- sum(bt_df$ret != 0, na.rm = TRUE)
cat("Backtest completato:\n")
cat("  Mesi totali   :", nrow(bt_df), "\n")
cat("  Con posizione :", n_traded, "\n")
cat("  No-trade      :", nrow(bt_df) - n_traded, "\n")
cat("  LONG  cluster-mesi:", sum(bt_df$n_long_cl,  na.rm=TRUE),
    "| mean ret:", round(mean(bt_df$mean_ret_long,  na.rm=TRUE)*100, 3), "%\n")
cat("  SHORT cluster-mesi:", sum(bt_df$n_short_cl, na.rm=TRUE),
    "| mean ret raw:", round(mean(bt_df$mean_ret_short, na.rm=TRUE)*100, 3), "%\n\n")


###########################################################################
# 16. PERFORMANCE FINALE
###########################################################################

perf_label <- paste0(
  CURRENT_RUN, "_Hybrid_L", best_cfg$long_classes,
  "_S", best_cfg$short_classes,
  "_tau", format(best_cfg$tau, trim = TRUE)
)
perf_df <- calc_performance(bt_df, label=perf_label)
if (is.null(perf_df)) {
  cat("WARN: nessun risultato valido.\n")
} else {
  cat("============================================\n")
  cat("  RISULTATO FINALE —", CURRENT_CONFIG, "| tau =", CURRENT_TAU, "| Hybrid\n")
  cat("============================================\n")
  cat(sprintf("  Mean Ann. Ret.  : %7.3f%%\n", perf_df$mean_ret_pct))
  cat(sprintf("  Total Return    : %7.2f%%\n", perf_df$total_ret_pct))
  cat(sprintf("  Sharpe Ratio    : %7.4f\n",   perf_df$sharpe))
  cat(sprintf("  Sortino Ratio   : %7.4f\n",   perf_df$sortino))
  cat(sprintf("  Max Drawdown    : %7.2f%%\n", perf_df$max_dd_pct))
  cat(sprintf("  Calmar Ratio    : %7.4f\n",   perf_df$calmar))
  cat(sprintf("  Profit Factor   : %7.4f\n",   perf_df$profit_factor))
  cat(sprintf("  %% No-trade      : %7.1f%%\n", perf_df$pct_no_trade))
  cat(sprintf("  N mesi test     : %7d\n",      perf_df$n_months))
  cat("============================================\n\n")
}

cat("Durata totale:", round(difftime(Sys.time(), t_start, units="mins"), 1), "min\n\n")


###########################################################################
# 17. SALVATAGGIO OUTPUT
###########################################################################

if (!is.null(perf_df) && nrow(bt_df) > 0) {
  saveRDS(perf_df,    out_file_perf)
  write.csv(perf_df, sub("\\.rds$",".csv",out_file_perf), row.names=FALSE)
  saveRDS(bt_df,      out_file_bt)
  write.csv(bt_df,   sub("\\.rds$",".csv",out_file_bt),   row.names=FALSE)
  if (!is.null(dnn_table7)) {
    saveRDS(dnn_table7, out_file_t7)
    write.csv(dnn_table7, sub("\\.rds$",".csv",out_file_t7), row.names=FALSE)
  }
  if (!is.null(dnn_table8)) {
    saveRDS(dnn_table8, out_file_t8)
    write.csv(dnn_table8, sub("\\.rds$",".csv",out_file_t8), row.names=FALSE)
  }
  cat("Output salvati in:", OUT_DIR, "\n")
  cat("  -", basename(out_file_perf), "\n")
  cat("  -", basename(out_file_bt),   "\n")
  if (!is.null(dnn_table7)) cat("  -", basename(out_file_t7), "\n")
  if (!is.null(dnn_table8)) cat("  -", basename(out_file_t8), "\n")
  if (file.exists(out_file_tune)) cat("  -", basename(out_file_tune), "\n")
} else {
  cat("WARN: nessun risultato valido — output NON salvati.\n")
}

cat("\n=====================================================\n")
cat("  SUPERVISED_DNN_C7m1.1_noTau v8 COMPLETATO\n")
cat("=====================================================\n\n")

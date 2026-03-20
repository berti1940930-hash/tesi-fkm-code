###########################################################################
###########################################################################
#############################     TESI       ##############################
###########################################################################
###########################################################################




#####################        PACCHETTI        #############################
###########################################################################
library(stats)
library(fclust)
library(data.table)
library(lubridate)
library(tidyr)
library(dplyr)
library(torch)
library(parallel)
library(foreach)
library(doParallel)

###########################################################################
# CONFIGURAZIONE HARDWARE – MacBook Air M4 (10-core CPU, 16 GB RAM)
###########################################################################
# Lascia 2 core all'OS; usa gli 8 rimanenti per FKM parallelo
N_CORES <- min(detectCores() - 2L, 8L)
cat("Core disponibili:", detectCores(), "| Core usati:", N_CORES, "\n")

OUT_DIR <- "/Users/macbookpro13/Desktop/TESI copy/05_Outputs"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cat("Output dir:", OUT_DIR, "\n")


#####################         DATI            #############################
###########################################################################

data <- read.csv("/users/macbookpro13/Desktop/TESI/Dati Vitale/firms_characteristics_3.csv")
data2 <- read.csv("/users/macbookpro13/Desktop/TESI/Dati Vitale/monthly_prices.csv")
tail(data)


#####################    Modifica DATI        #############################
###########################################################################

# Converti le date in formato Date
data$date <- as.Date(data$DATE)
data2$date <- as.Date(data2$date)

# Ordina per data
data <- data[order(data$permno, data$date), ]
data2 <- data2[order(data2$date), ]

# Verifica
str(data)
str(data2)
data <- data %>% select(-DATE) 
###########################################################################
#####################    CALCOLO MOMENTUM     #############################
###########################################################################

# TRASFORMA PRICES DA WIDE A LONG


prices_long <- data2 %>%
  pivot_longer(
    cols = -date,
    names_to = "permno",
    values_to = "price"
  ) %>%
  mutate(
    permno = as.integer(gsub("^X", "", permno))  # <-- FIX
  ) %>%
  filter(!is.na(permno)) %>%                     # <-- safety
  filter(!is.na(price)) %>%
  arrange(permno, date)


# CALCOLA I RITORNI MENSILI
prices_long <- prices_long %>%
  group_by(permno) %>%
  mutate(ret = (price / lag(price)) - 1) %>%
  ungroup()

# CALCOLA LE 48 MOMENTUM FEATURES (come in Vitale)
prices_long <- prices_long %>%
  group_by(permno) %>%
  arrange(date) %>%
  mutate(
    log1pr = log1p(ret),
    mom_1 = lag(ret, 1)
  ) %>%
  ungroup()

# Per mom_2 a mom_48 (cumulativi da t-i a t-2)
cat("\n[BLOCCO 1/7] Calcolo momentum mom_2..mom_48 (loop 47 iter)...\n")
t0_mom <- Sys.time()
for(i in 2:48) {
  col_name <- paste0("mom_", i)
  if (i %% 12 == 0) cat("  mom_", i, "/ 48 —", format(Sys.time(), "%H:%M:%S"), "\n")
  
  prices_long <- prices_long %>%
    group_by(permno) %>%
    arrange(date) %>%
    mutate(
      !!col_name := expm1(
        zoo::rollsum(lag(log1pr, 2), k = i-1, fill = NA, align = "right")
      )
    ) %>%
    ungroup()
}
cat("  ✓ Momentum completato in",
    round(difftime(Sys.time(), t0_mom, units = "secs")), "sec\n")
saveRDS(prices_long, file.path(OUT_DIR, "prices_long.rds"))

# Verifica
head(prices_long[, c("permno", "date", "ret", "mom_1", "mom_2", "mom_12")])

###########################################################################
#####################    MERGE DATASETS       #############################
###########################################################################



# (B) merge base (ancora senza tagliare a 2004)
final_df <- merge(prices_long, data, by = c("permno", "date"), all = FALSE)

# (C) definisci colonne
mom_cols  <- paste0("mom_", 1:48)
char_cols <- setdiff(names(data), c("permno", "date"))

# -----------------------------------------------------------------------
# (D) FILTRO VITALE: no missing su *tutto l’orizzonte 2000-01 .. 2020-12*
# -----------------------------------------------------------------------

# merge completo su tutto il periodo
merged_full <- merge(prices_long, data, by = c("permno", "date"), all = FALSE)

# colonne characteristics (come Python: fc.columns senza permno/date)
char_cols <- setdiff(names(data), c("permno", "date"))

# filtro periodo 2000-01 .. 2020-12 (come notebook)
analysis_fc <- merged_full %>%
  filter(date >= as.Date("2000-01-01") & date <= as.Date("2020-12-31"))

# permno puliti: quelli con ZERO NA nelle firm characteristics (nessun vincolo sui mesi!)
clean_permnos_fc <- analysis_fc %>%
  group_by(permno) %>%
  summarise(has_na = any(if_any(all_of(char_cols), is.na)), .groups = "drop") %>%
  filter(!has_na) %>%
  pull(permno)

cat("permno senza NA nelle chars (atteso 285):", length(clean_permnos_fc), "\n")

# ora dataset finale 2004-02 .. 2020-12 con quei permno
final_df <- merged_full %>%
  filter(date >= as.Date("2004-02-01") & date <= as.Date("2020-12-31")) %>%
  filter(permno %in% clean_permnos_fc)

# colonne momentum + chars + ret come in notebook
mom_cols <- paste0("mom_", 1:48)
cols_to_check <- c("ret", mom_cols, char_cols)

# drop righe con NA nelle colonne critiche (come final_df.dropna(subset=...))
final_df <- final_df %>%
  filter(complete.cases(across(all_of(cols_to_check)))) %>%
  arrange(permno, date)

cat("Finale: permno =", dplyr::n_distinct(final_df$permno),
    " months =", dplyr::n_distinct(format(final_df$date, "%Y-%m")),
    " rows =", nrow(final_df), "\n")




# -----------------------------------------------------------------------
# (E) tagli a 2004-02 .. 2020-12 (perché servono 48 mom)
# -----------------------------------------------------------------------
final_df <- final_df %>%
  filter(date >= as.Date("2004-02-01") & date <= as.Date("2020-12-31"))

# (F) rimuovi righe con NA nelle colonne critiche
cols_to_check <- c("ret", mom_cols, char_cols)

final_df <- final_df %>%
  filter(complete.cases(across(all_of(cols_to_check)))) %>%
  arrange(permno, date)

cat("Finale: permno =", dplyr::n_distinct(final_df$permno),
    " months =", dplyr::n_distinct(format(final_df$date, "%Y-%m")),
    " rows =", nrow(final_df), "\n")



###########################################################################
#####################    PREPARAZIONE PCA     #############################
###########################################################################
X_cols <- c(mom_cols, char_cols)

# ym (year-month) come ID mese
final_df <- final_df %>% mutate(ym = format(date, "%Y-%m"))
saveRDS(final_df, file.path(OUT_DIR, "final_df_pre_pca.rds"))
X_cols_num <- X_cols[sapply(final_df[, X_cols], is.numeric)]
length(X_cols_num)   # sanity

# ---- 1) calcola K_t per mese (data-driven) ----
K_by_month <- final_df %>%
  group_by(ym) %>%
  group_modify(~{
    dfm <- .x
    X  <- as.matrix(dfm[, X_cols_num])
    Xs <- scale(X)
    
    keep <- !apply(Xs, 2, function(z) all(is.na(z)))
    Xs <- Xs[, keep, drop = FALSE]
    Xs[is.na(Xs)] <- 0
    
    p  <- prcomp(Xs, center = FALSE, scale. = FALSE)
    ev <- (p$sdev^2) / sum(p$sdev^2)
    
    K_t <- which(cumsum(ev) >= 0.99)[1]
    K_t <- min(K_t, nrow(Xs) - 1, ncol(Xs))  # cap tecnico, non “metodologico”
    
    
    data.frame(K_t = K_t)
    
  }) %>%
  ungroup()


# ---- 2) PCA mensile finale con K fisso ----
cat("\n[BLOCCO 2/7] PCA mensile adattiva (203 mesi × 285 titoli × 122 var)...\n")
t0_pca <- Sys.time()
var_target <- 0.99

pca_list <- final_df %>%
  mutate(ym = format(date, "%Y-%m")) %>%
  group_by(ym) %>%
  group_modify(~{
    dfm <- .x
    
    X  <- as.matrix(dfm[, X_cols_num])
    mu <- colMeans(X, na.rm = TRUE)
    sd <- apply(X, 2, sd, na.rm = TRUE)
    
    keep <- is.finite(sd) & (sd > 0)
    
    p_raw  <- ncol(X)
    p_keep <- sum(keep)
    
    Xc <- sweep(X[, keep, drop=FALSE], 2, mu[keep], "-")
    Xs <- sweep(Xc, 2, sd[keep], "/")
    Xs[!is.finite(Xs)] <- 0
    
    p  <- prcomp(Xs, center = FALSE, scale. = FALSE)
    ev <- (p$sdev^2) / sum(p$sdev^2)
    
    p_t <- which(cumsum(ev) >= var_target)[1]
    p_t <- min(p_t, nrow(Xs) - 1, ncol(Xs))
    
    scores <- p$x[, 1:p_t, drop = FALSE]
    colnames(scores) <- paste0("PC", 1:p_t)
    
    out <- bind_cols(dfm[, c("permno","date","ret")], as.data.frame(scores))
    out$ym     <- unique(dfm$ym)[1]
    out$p_t    <- p_t
    out$p_raw  <- p_raw
    out$p_keep <- p_keep
    out
    
  }) %>%
  ungroup() %>%
  arrange(permno, date)

cat("  ✓ PCA completata in",
    round(difftime(Sys.time(), t0_pca, units = "secs")), "sec\n")
cat("Dimensioni pca_list:", dim(pca_list), "\n")
# ---- CHECK p_t per MESE (non per riga) ----
p_by_month <- pca_list %>% distinct(ym, p_t)

cat("\nDistribuzione p_t per mese (corretta):\n")
print(summary(p_by_month$p_t))
cat("Mean p_t (per mese):", mean(p_by_month$p_t), "\n")
cat("N mesi:", nrow(p_by_month), "\n")

###########################################################################
########################## SALVATAGGIO DATI PCA ###########################
###########################################################################


# 1) PANEL PCA
saveRDS(pca_list, file.path(OUT_DIR, "pca_panel_99pct.rds"))


pca_list <- readRDS(file.path(OUT_DIR, "pca_panel_99pct.rds"))





###########################################################################
#####################    FUZZY CLUSTERING     #############################
##########################    GRID    #####################################
###########################################################################

######### FASE PRELIMINARE ########

# Estrazioni centroidi

get_centers <- function(fit) {
  if (!is.null(fit$H)) return(fit$H)
  if (!is.null(fit$C)) return(fit$C)
  if (!is.null(fit$centers)) return(fit$centers)
  stop("Non trovo i centroidi nel fit: stampa names(fit)")
}

# Calcolo metrica: PC, Partition Coefficient

fpc <- function(U) mean(rowSums(U^2))

# Calcolo metrica: MPC, Modified Partition Coefficient. Importante per confrontare partizioni con cluster diversi

mpc <- function(U) {
  C <- ncol(U)
  f <- fpc(U)
  (f - 1/C) / (1 - 1/C)
}

# Calcolo metrica: PE, Partition Entropy

pe <- function(U) {
  U2 <- pmax(U, 1e-12)
  -mean(rowSums(U2 * log(U2)))
}

# Calcolo metrica: XB, Xie and Beni Index

xb_index <- function(X, U, V, m) {
  n <- nrow(X); C <- ncol(U)
  Um <- U^m
  
  num <- 0
  for (c in 1:C) {
    diff <- sweep(X, 2, V[c,], "-")
    d2   <- rowSums(diff^2)
    num  <- num + sum(Um[, c] * d2)
  }
  
  
  
  mind2 <- Inf
  for (c in 1:(C-1)) for (k in (c+1):C) {
    d2 <- sum((V[c,] - V[k,])^2)
    if (d2 < mind2) mind2 <- d2
  }
  
  num / (n * mind2)
}

# Griglie e mesi

# Numero Cluster
C_grid <- 2:10

# Fuzziness
m_grid <- c(1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4)

months_all <- sort(unique(pca_list$ym))
months_use <- sample(months_all, size = 40)  # test veloce, PRELIMINARE
months_use <- months_all                       # full

RS <- 3          # 3 inizializzazioni casuali nella fase di screening
set.seed(123)

if (!file.exists(file.path(OUT_DIR, "grid_screen_RS3.rds"))) {
  
  cat("\n[BLOCCO 3/7] FKM grid screening —",
      length(months_use), "mesi ×", length(C_grid) * length(m_grid),
      "config × RS=3 — su", N_CORES, "core...\n")
  t0_screen <- Sys.time()
  
  cl <- makeCluster(N_CORES)
  registerDoParallel(cl)
  clusterExport(cl, c("get_centers", "fpc", "mpc", "pe", "xb_index",
                      "pca_list", "C_grid", "m_grid", "RS"))
  clusterEvalQ(cl, { library(fclust); library(dplyr) })
  
  res_list <- foreach(
    mm = months_use,
    .packages  = c("fclust", "dplyr"),
    .errorhandling = "pass"
  ) %dopar% {
    
    dfm <- pca_list %>% filter(ym == mm)
    
    pc_cols <- names(dfm)[grepl("^PC[0-9]+$", names(dfm))]
    pc_cols <- pc_cols[order(as.integer(sub("^PC","", pc_cols)))]
    pc_cols <- pc_cols[!sapply(dfm[, pc_cols, drop = FALSE], function(z) all(is.na(z)))]
    
    X <- as.matrix(dfm[, pc_cols, drop = FALSE])
    if (anyNA(X)) return(NULL)
    
    out <- vector("list", length(C_grid) * length(m_grid))
    idx <- 0
    
    for (C in C_grid) for (m in m_grid) {
      idx <- idx + 1
      tryCatch({
        fit <- FKM(X, k = C, m = m, RS = RS)
        U   <- fit$U
        V   <- get_centers(fit)
        
        out[[idx]] <- data.frame(
          ym       = mm,
          C        = C,
          m        = m,
          XB       = xb_index(X, U, V, m),
          FPC      = fpc(U),
          MPC      = mpc(U),
          PE       = pe(U),
          u_median = median(apply(U, 1, max))
        )
      }, error = function(e) NULL)
    }
    
    dplyr::bind_rows(out)
  }
  
  stopCluster(cl)
  
  res_screen <- dplyr::bind_rows(res_list)
  saveRDS(res_screen, file.path(OUT_DIR, "grid_screen_RS3.rds"))
  cat("  ✓ Grid screening completato in",
      round(difftime(Sys.time(), t0_screen, units = "mins"), 1), "min\n")
  
} else {
  cat("\n[BLOCCO 3/7] Grid screening già calcolato — carico da disco\n")
  res_screen <- readRDS(file.path(OUT_DIR, "grid_screen_RS3.rds"))
}


summary_all <- res_screen %>%
  group_by(C, m) %>%
  summarise(
    n_months = n(),
    # livello centrale
    XB_med   = median(XB, na.rm=TRUE),
    MPC_med  = median(MPC, na.rm=TRUE),
    FPC_med  = median(FPC, na.rm=TRUE),
    PE_med   = median(PE, na.rm=TRUE),
    u_med    = median(u_median, na.rm=TRUE),
    # dispersione (robustezza)
    XB_iqr   = IQR(XB, na.rm=TRUE),
    MPC_iqr  = IQR(MPC, na.rm=TRUE),
    PE_iqr   = IQR(PE, na.rm=TRUE),
    u_iqr    = IQR(u_median, na.rm=TRUE),
    .groups  = "drop"
  ) %>%
  arrange(C, m)

print(summary_all, n=90)

# -------------------------------------------------------------------
# COMMENTO AI RISULTATI DELLO SCREENING FUZZY (summary_all)
#
# La tabella summary_all riassume, per ciascuna combinazione di:
#   - C = numero di cluster
#   - m = parametro di fuzziness
# i valori mediani mensili degli indici di qualità del clustering
# (XB, MPC, FPC, PE, u_med) e la loro dispersione intertemporale (IQR).
#
# LETTURA GENERALE
# - m è il parametro che incide di più sulla qualità del clustering.
# - Per quasi tutti i valori di C, le configurazioni con m = 1.1
#   mostrano i risultati più stabili e statisticamente sensati:
#     * XB più basso
#     * MPC e FPC più alti
#     * PE più basso
#     * u_med molto elevato
# - All'aumentare di m, la struttura dei cluster diventa progressivamente
#   più sfumata: MPC e FPC diminuiscono, PE aumenta e u_med si riduce.
#
# EVIDENZA DI DEGENERAZIONE
# - Per valori elevati di m (in particolare da circa m >= 1.8),
#   molte configurazioni convergono verso una situazione quasi uniforme:
#     * MPC ~ 0
#     * FPC ~ 1/C
#     * PE ~ log(C)
#     * u_med ~ 1/C
# - Questo comportamento segnala che i titoli vengono assegnati ai cluster
#   con membership quasi uguali, quindi il clustering perde contenuto
#   informativo e diventa poco utile economicamente.
#
# IMPLICAZIONE PER LA SELEZIONE DELLE CONFIGURAZIONI
# - Poiché l'obiettivo finale non è massimizzare la "bellezza" statistica
#   del clustering, ma ottenere buone performance di trading out-of-sample,
#   questa tabella NON va usata per scegliere automaticamente il setup con
#   il miglior XB o MPC.
# - Gli indici di clustering servono soprattutto come filtro di ammissibilità:
#   consentono di scartare le configurazioni chiaramente degeneri o divergenti.
# - Tra le configurazioni ancora sensate dal punto di vista statistico,
#   la scelta finale deve essere guidata dalle metriche economiche della
#   strategia (rendimento medio, Sharpe ratio, drawdown, stabilità nel tempo,
#   numero di pair effettivamente tradabili, ecc.).
#
# INDICAZIONE PRATICA
# - Le configurazioni con m = 1.1 rappresentano il nucleo più solido.
# - Le configurazioni con m = 1.2 possono ancora essere considerate valide,
#   soprattutto se offrono vantaggi in termini di rendimento.
# - Le configurazioni con m = 1.3 sono da valutare con cautela, specie per
#   C elevati, perché mostrano già segnali di forte deterioramento.
# - Le configurazioni con m >= 1.4 vanno considerate borderline, mentre
#   quelle con m >= 1.8 sono in larga misura da escludere, salvo evidenza
#   economica molto forte e robusta.
# -------------------------------------------------------------------
--------------------------------------------------------------------------------
  
  # 
  '
ora salviamo le 10 configurazioni interessanti che non esplodono
# '

###########################################################################
###################  REFIT 10 CONFIGURAZIONI TOP  #########################
###########################################################################


configs <- data.frame(
  C = c(3,   4,   5,   6,   7,   8,   9,   10,  3,   4  ),
  m = c(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2)
)
# C=3..10 m=1.1: nucleo non degenere, copre tutta la granularità incluso C=10 (= Vitale K)
# C=3, C=4 m=1.2: test robustezza al fuzziness (unici valori ancora ammissibili per m=1.2)

if (!file.exists(file.path(OUT_DIR, "refit_10configs_RS20.rds"))) {
  
  RS_final      <- 20
  set.seed(42)
  months_sorted <- sort(unique(pca_list$ym))
  
  cat("\n[BLOCCO 4/7] FKM refit — 10 config × 203 mesi × RS=20 — su", N_CORES, "core...\n")
  t0_refit <- Sys.time()
  
  cl <- makeCluster(N_CORES)
  registerDoParallel(cl)
  clusterExport(cl, c("pca_list", "RS_final"))
  clusterEvalQ(cl, { library(fclust); library(dplyr) })
  
  refit_results <- list()
  
  for (i in seq_len(nrow(configs))) {
    
    C_i     <- configs$C[i]
    m_i     <- configs$m[i]
    label_i <- paste0("C", C_i, "_m", m_i)
    
    cat("  [", i, "/", nrow(configs), "] Fitting:", label_i, "— ", format(Sys.time(), "%H:%M:%S"), "\n")
    t0 <- Sys.time()
    
    month_fits <- foreach(
      mm             = months_sorted,
      .packages      = c("fclust", "dplyr"),
      .errorhandling = "pass"
    ) %dopar% {
      
      dfm <- pca_list %>% filter(ym == mm)
      
      pc_cols <- names(dfm)[grepl("^PC[0-9]+$", names(dfm))]
      pc_cols <- pc_cols[order(as.integer(sub("^PC", "", pc_cols)))]
      pc_cols <- pc_cols[!sapply(dfm[, pc_cols, drop = FALSE],
                                 function(z) all(is.na(z)))]
      
      X <- as.matrix(dfm[, pc_cols, drop = FALSE])
      valid_permnos <- dfm$permno[complete.cases(dfm[, pc_cols, drop = FALSE])]
      X <- X[complete.cases(X), , drop = FALSE]
      
      if (nrow(X) < C_i) return(NULL)
      
      fit <- tryCatch(
        FKM(X, k = C_i, m = m_i, RS = RS_final),
        error = function(e) NULL
      )
      
      if (is.null(fit)) return(NULL)
      
      list(
        permno = valid_permnos,
        U      = fit$U,
        assign = apply(fit$U, 1, which.max)
      )
    }
    
    names(month_fits)        <- months_sorted
    refit_results[[label_i]] <- month_fits
    
    cat("    ✓ completato in",
        round(difftime(Sys.time(), t0, units = "secs")), "sec\n")
  }
  
  stopCluster(cl)
  
  saveRDS(refit_results, file.path(OUT_DIR, "refit_10configs_RS20.rds"))
  saveRDS(final_df,      file.path(OUT_DIR, "final_df.rds"))
  cat("  ✓ Refit completato in",
      round(difftime(Sys.time(), t0_refit, units = "mins"), 1), "min\n")
  
} else {
  cat("\n[BLOCCO 4/7] FKM refit già calcolato — carico da disco\n")
  refit_results <- readRDS(file.path(OUT_DIR, "refit_10configs_RS20.rds"))
  months_sorted <- sort(unique(pca_list$ym))
}

# ---------------------------------  

###########################################################################
###############    STRATEGIA TRADING CLUSTERING-BASED  ###################
###################   SENZA RIMOZIONE OUTLIERS     #######################
#########################################################################

refit_results <- readRDS(file.path(OUT_DIR, "refit_10configs_RS20.rds"))
final_df      <- readRDS(file.path(OUT_DIR, "final_df.rds"))

if (!"ym" %in% names(final_df)) {
  final_df <- final_df %>% mutate(ym = format(date, "%Y-%m"))
}

months_sorted <- sort(unique(final_df$ym))


# FUNZIONE CORE


run_strategy <- function(config_fits, final_df, months_sorted) {
  
  results <- lapply(seq_along(months_sorted), function(t_idx) {
    
    if (t_idx == length(months_sorted)) return(NULL)
    
    t_month  <- months_sorted[t_idx]
    t1_month <- months_sorted[t_idx + 1]
    
    fit_t <- config_fits[[t_month]]
    if (is.null(fit_t)) return(NULL)
    
    # segnale = ret al mese t (= r_t, il ritorno appena concluso)
    # coerente con la short-term reversal classica: sort su r_t, predici r_{t+1}
    dfm_t <- final_df %>%
      filter(ym == t_month) %>%
      select(permno, signal = ret)   # rinomina per evitare conflitto con ret di t+1
    
    dfm_t1 <- final_df %>%
      filter(ym == t1_month) %>%
      select(permno, ret)
    
    trade_df <- data.frame(
      permno  = fit_t$permno,
      cluster = fit_t$assign
    ) %>%
      inner_join(dfm_t,  by = "permno") %>%
      inner_join(dfm_t1, by = "permno") %>%
      filter(!is.na(signal), !is.na(ret))
    
    if (nrow(trade_df) == 0) return(NULL)
    
    # cluster totali presenti questo mese
    n_clusters_total <- length(unique(trade_df$cluster))
    
    # ----------------------------------------------------------------
    # STEP 1: costruisci tutte le coppie e calcola sigma GLOBALE
    # ----------------------------------------------------------------
    all_diffs  <- numeric(0)
    pair_store <- list()
    
    for (cl in sort(unique(trade_df$cluster))) {
      
      sub     <- trade_df %>% filter(cluster == cl) %>% arrange(signal)
      n_cl    <- nrow(sub)
      n_pairs <- floor(n_cl / 2)
      if (n_pairs == 0) next
      
      idx_long  <- 1:n_pairs
      idx_short <- rev((n_cl - n_pairs + 1):n_cl)
      
      diffs <- sub$signal[idx_short] - sub$signal[idx_long]
      
      all_diffs <- c(all_diffs, diffs)
      
      pair_store[[as.character(cl)]] <- list(
        ret_long  = sub$ret[idx_long],
        ret_short = sub$ret[idx_short],
        diffs     = diffs
      )
    }
    
    if (length(all_diffs) == 0) return(NULL)
    
    # sigma cross-sectionale globale (Vitale-compliant)
    sigma <- sd(all_diffs, na.rm = TRUE)
    if (is.na(sigma) || sigma == 0) return(NULL)
    
    # ----------------------------------------------------------------
    # STEP 2: filtra coppie con diff > sigma, aggrega EW per COPPIA
    # ----------------------------------------------------------------
    pair_rets         <- numeric(0)
    n_pairs_pre       <- length(all_diffs)
    n_pairs_post      <- 0
    n_clusters_traded <- 0
    
    for (cl in names(pair_store)) {
      
      pl    <- pair_store[[cl]]
      valid <- which(pl$diffs > sigma)
      if (length(valid) == 0) next
      
      n_clusters_traded <- n_clusters_traded + 1
      
      coppia_rets  <- pl$ret_long[valid] - pl$ret_short[valid]
      pair_rets    <- c(pair_rets, coppia_rets)
      n_pairs_post <- n_pairs_post + length(valid)
    }
    
    strat_ret <- if (length(pair_rets) == 0) 0 else mean(pair_rets)
    
    list(
      ym                 = t_month,
      ret                = strat_ret,
      n_pairs_pre        = n_pairs_pre,
      n_pairs_post       = n_pairs_post,
      n_clusters_total   = n_clusters_total,
      n_clusters_pairs   = length(pair_store),
      n_clusters_traded  = n_clusters_traded
    )
  })
  
  results <- Filter(Negate(is.null), results)
  
  data.frame(
    ym                = sapply(results, `[[`, "ym"),
    ret               = sapply(results, `[[`, "ret"),
    n_pairs_pre       = sapply(results, `[[`, "n_pairs_pre"),
    n_pairs_post      = sapply(results, `[[`, "n_pairs_post"),
    n_clusters_total  = sapply(results, `[[`, "n_clusters_total"),
    n_clusters_pairs  = sapply(results, `[[`, "n_clusters_pairs"),
    n_clusters_traded = sapply(results, `[[`, "n_clusters_traded")
  )
}


# FUNZIONE PERFORMANCE


calc_performance <- function(df_rets, label = "") {

  rets       <- df_rets$ret
  rets_clean <- rets[!is.na(rets)]

  # --- rendimenti di base ---
  mean_ret_m  <- mean(rets_clean)
  mean_ret_y  <- mean_ret_m * 12
  total_ret   <- prod(1 + rets_clean) - 1
  sd_y        <- sd(rets_clean) * sqrt(12)          # deviazione standard annualizzata

  # --- Sharpe Ratio ---
  sharpe <- if (sd(rets_clean) == 0) NA else mean_ret_y / sd_y

  # --- Sortino Ratio (semi-deviazione standard, target = 0, su tutti i rendimenti) ---
  down_dev_y   <- sqrt(mean(pmin(rets_clean, 0)^2)) * sqrt(12)
  sortino      <- if (is.na(down_dev_y) || down_dev_y == 0) NA else mean_ret_y / down_dev_y

  # --- Maximum Drawdown ---
  cum    <- cumprod(1 + rets_clean)
  max_dd <- min((cum - cummax(cum)) / cummax(cum))

  # --- Calmar Ratio ---
  calmar <- if (max_dd == 0) NA else mean_ret_y / abs(max_dd)

  # --- Profit Factor ---
  gross_profit <- sum(rets_clean[rets_clean > 0])
  gross_loss   <- abs(sum(rets_clean[rets_clean < 0]))
  profit_factor <- if (gross_loss == 0) NA else gross_profit / gross_loss

  # --- Anni profittevoli / non profittevoli ---
  if ("ym" %in% names(df_rets)) {
    ym_clean    <- df_rets$ym[!is.na(df_rets$ret)]
    years       <- substr(ym_clean, 1, 4)
    annual_rets <- tapply(rets_clean, years, function(r) prod(1 + r) - 1)
    profitable_years   <- sum(annual_rets > 0)
    unprofitable_years <- sum(annual_rets <= 0)
  } else {
    profitable_years   <- NA
    unprofitable_years <- NA
  }

  # --- % mesi senza trade ---
  pct_no_trade <- if ("n_pairs_post" %in% names(df_rets))
    mean(df_rets$n_pairs_post == 0) * 100 else
      mean(is.na(rets)) * 100

  data.frame(
    config             = label,
    mean_ret_pct       = round(mean_ret_y    * 100, 4),
    sd_ret_pct         = round(sd_y          * 100, 4),
    total_ret_pct      = round(total_ret     * 100, 2),
    sharpe             = round(sharpe,               4),
    sortino            = round(sortino,              4),
    max_dd_pct         = round(max_dd        * 100, 2),
    calmar             = round(calmar,               4),
    profit_factor      = round(profit_factor,        4),
    profitable_years   = profitable_years,
    unprofitable_years = unprofitable_years,
    pct_no_trade       = round(pct_no_trade,         1),
    n_months           = length(rets_clean)
  )
}


cat("\n[BLOCCO 5/7] Strategia FKM base (10 config × 203 mesi)...\n")
t0_strat <- Sys.time()

all_rets <- list()
all_perf <- list()

for (label in names(refit_results)) {
  
  cat("Strategia:", label, "...\n")
  
  df_rets <- run_strategy(
    config_fits   = refit_results[[label]],
    final_df      = final_df,
    months_sorted = months_sorted
  )
  
  perf <- calc_performance(df_rets, label = label)
  
  all_rets[[label]] <- df_rets
  all_perf[[label]] <- perf
  
  cat("  Mean (ann.):", perf$mean_ret_pct, "%",
      "| Sharpe:", perf$sharpe,
      "| MaxDD:", perf$max_dd_pct, "%",
      "| No-trade:", perf$pct_no_trade, "%\n")
}


# BENCHMARK: SHORT-TERM REVERSAL (long losers, short winners)


cat("Benchmark: Short-term Reversal...\n")

reversal_rets <- sapply(seq_along(months_sorted)[-length(months_sorted)],
                        function(t_idx) {
                          
                          t_month  <- months_sorted[t_idx]
                          t1_month <- months_sorted[t_idx + 1]
                          
                          # segnale = ret al mese t (r_t), outcome = ret al mese t+1
                          # short-term reversal classica: 1 mese di gap, non 2
                          dfm_t  <- final_df %>% filter(ym == t_month)  %>% select(permno, signal = ret)
                          dfm_t1 <- final_df %>% filter(ym == t1_month) %>% select(permno, ret)
                          
                          dfm <- inner_join(dfm_t, dfm_t1, by = "permno") %>%
                            filter(!is.na(signal), !is.na(ret))
                          
                          if (nrow(dfm) < 10) return(NA)
                          
                          q <- quantile(dfm$signal, probs = c(0.2, 0.8), na.rm = TRUE)
                          mean(dfm$ret[dfm$signal <= q[1]]) - mean(dfm$ret[dfm$signal >= q[2]])
                        })

reversal_df   <- data.frame(
  ym  = months_sorted[-length(months_sorted)],
  ret = reversal_rets
)
reversal_perf <- calc_performance(reversal_df, label = "Reversal_Benchmark")


# TABELLA FINALE


perf_final <- bind_rows(c(all_perf, list(reversal_perf)))
print(perf_final)


# SALVATAGGIO


saveRDS(all_rets,   file.path(OUT_DIR, "strategy_returns.rds"))
saveRDS(perf_final, file.path(OUT_DIR, "performance_table.rds"))
write.csv(perf_final, file.path(OUT_DIR, "performance_table.csv"), row.names = FALSE)

cat("  ✓ Strategia base completata in",
    round(difftime(Sys.time(), t0_strat, units = "secs")), "sec\n")


# --------------------------------------------------------------------------


###########################################################################
###############    STRATEGIA TRADING CLUSTERING-BASED  ###################
###################   con RIMOZIONE OUTLIERS     #######################
#########################################################################
###########################################################################
#####################    STRATEGIA CON SOGLIA TAU  #######################
###########################################################################

refit_results <- readRDS(file.path(OUT_DIR, "refit_10configs_RS20.rds"))
final_df      <- readRDS(file.path(OUT_DIR, "final_df.rds"))

if (!"ym" %in% names(final_df)) {
  final_df <- final_df %>% mutate(ym = format(date, "%Y-%m"))
}

months_sorted <- sort(unique(final_df$ym))

# Soglie tau (0 = baseline senza filtro)
tau_grid <- c(0, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)

###########################################################################
# FUNZIONE CORE CON TAU
###########################################################################

run_strategy_tau <- function(config_fits, final_df, months_sorted, tau = 0) {
  
  results <- lapply(seq_along(months_sorted), function(t_idx) {
    
    if (t_idx == length(months_sorted)) return(NULL)
    
    t_month  <- months_sorted[t_idx]
    t1_month <- months_sorted[t_idx + 1]
    
    fit_t <- config_fits[[t_month]]
    if (is.null(fit_t)) {
      return(list(ym=t_month, ret=0, n_pairs_pre=0, n_pairs_post=0,
                  n_clusters_total=0, n_clusters_traded=0,
                  n_stocks_kept=0, n_stocks_filtered=0))
    }
    
    # segnale = ret al mese t (r_t); rinominato 'signal' per evitare conflitto con ret t+1
    dfm_t  <- final_df[final_df$ym == t_month,  c("permno","ret")]
    names(dfm_t)[names(dfm_t) == "ret"] <- "signal"
    dfm_t1 <- final_df[final_df$ym == t1_month, c("permno","ret")]
    
    U       <- fit_t$U
    n_stock <- nrow(U)
    
    tau_assign <- sapply(1:n_stock, function(i) {
      u_i   <- U[i, ]
      above <- which(u_i >= tau)
      if (length(above) == 0) return(NA)
      above[which.max(u_i[above])]
    })
    
    base_df <- merge(
      merge(
        data.frame(permno=fit_t$permno, cluster_tau=tau_assign),
        dfm_t,  by="permno"
      ),
      dfm_t1, by="permno"
    )
    base_df <- base_df[!is.na(base_df$signal) & !is.na(base_df$ret), ]
    
    if (nrow(base_df) == 0) {
      return(list(ym=t_month, ret=0, n_pairs_pre=0, n_pairs_post=0,
                  n_clusters_total=0, n_clusters_traded=0,
                  n_stocks_kept=0, n_stocks_filtered=0))
    }
    
    n_total_candidates <- nrow(base_df)
    trade_df           <- base_df[!is.na(base_df$cluster_tau), ]
    n_stocks_kept      <- nrow(trade_df)
    n_stocks_filtered  <- n_total_candidates - n_stocks_kept
    
    if (n_stocks_kept == 0) {
      return(list(ym=t_month, ret=0, n_pairs_pre=0, n_pairs_post=0,
                  n_clusters_total=0, n_clusters_traded=0,
                  n_stocks_kept=0, n_stocks_filtered=n_total_candidates))
    }
    
    clusters_present <- sort(unique(trade_df$cluster_tau))
    n_clusters_total <- length(clusters_present)
    
    all_diffs  <- numeric(0)
    pair_store <- list()
    
    for (cl in clusters_present) {
      
      sub     <- trade_df[trade_df$cluster_tau == cl, ]
      sub     <- sub[order(sub$signal), ]
      n_cl    <- nrow(sub)
      n_pairs <- floor(n_cl / 2)
      if (n_pairs == 0) next
      
      idx_low  <- 1:n_pairs
      idx_high <- rev((n_cl - n_pairs + 1):n_cl)
      
      diffs <- sub$signal[idx_high] - sub$signal[idx_low]
      
      all_diffs <- c(all_diffs, diffs)                   # ← RIGA CHIAVE
      
      pair_store[[as.character(cl)]] <- list(
        ret_long  = sub$ret[idx_low],
        ret_short = sub$ret[idx_high],
        diffs     = diffs
      )
    }
    
    if (length(all_diffs) == 0) {
      return(list(ym=t_month, ret=0, n_pairs_pre=0, n_pairs_post=0,
                  n_clusters_total=n_clusters_total, n_clusters_traded=0,
                  n_stocks_kept=n_stocks_kept, n_stocks_filtered=n_stocks_filtered))
    }
    
    sigma <- sd(all_diffs, na.rm=TRUE)
    if (is.na(sigma) || sigma == 0) {
      return(list(ym=t_month, ret=0, n_pairs_pre=length(all_diffs), n_pairs_post=0,
                  n_clusters_total=n_clusters_total, n_clusters_traded=0,
                  n_stocks_kept=n_stocks_kept, n_stocks_filtered=n_stocks_filtered))
    }
    
    pair_rets         <- numeric(0)
    n_pairs_pre       <- length(all_diffs)
    n_pairs_post      <- 0
    n_clusters_traded <- 0
    
    for (cl in names(pair_store)) {
      pl    <- pair_store[[cl]]
      valid <- which(pl$diffs > sigma)
      if (length(valid) == 0) next
      n_clusters_traded <- n_clusters_traded + 1
      pair_rets    <- c(pair_rets, pl$ret_long[valid] - pl$ret_short[valid])
      n_pairs_post <- n_pairs_post + length(valid)
    }
    
    strat_ret <- if (length(pair_rets) == 0) 0 else mean(pair_rets)
    
    list(ym=t_month, ret=strat_ret,
         n_pairs_pre=n_pairs_pre, n_pairs_post=n_pairs_post,
         n_clusters_total=n_clusters_total, n_clusters_traded=n_clusters_traded,
         n_stocks_kept=n_stocks_kept, n_stocks_filtered=n_stocks_filtered)
  })
  
  results <- Filter(Negate(is.null), results)
  if (length(results) == 0) return(NULL)
  
  data.frame(
    ym                = sapply(results, `[[`, "ym"),
    ret               = sapply(results, `[[`, "ret"),
    n_pairs_pre       = sapply(results, `[[`, "n_pairs_pre"),
    n_pairs_post      = sapply(results, `[[`, "n_pairs_post"),
    n_clusters_total  = sapply(results, `[[`, "n_clusters_total"),
    n_clusters_traded = sapply(results, `[[`, "n_clusters_traded"),
    n_stocks_kept     = sapply(results, `[[`, "n_stocks_kept"),
    n_stocks_filtered = sapply(results, `[[`, "n_stocks_filtered")
  )
}

###########################################################################
# FUNZIONE PERFORMANCE
###########################################################################

calc_performance <- function(df_rets, label = "") {
  
  if (is.null(df_rets)) return(NULL)
  
  rets       <- df_rets$ret
  rets_clean <- rets[!is.na(rets)]
  
  if (length(rets_clean) == 0) return(NULL)
  
  mean_ret_m  <- mean(rets_clean)
  mean_ret_y  <- mean_ret_m * 12
  total_ret   <- prod(1 + rets_clean) - 1
  sharpe      <- if (sd(rets_clean) == 0) NA else
    mean_ret_y / (sd(rets_clean) * sqrt(12))
  
  cum    <- cumprod(1 + rets_clean)
  max_dd <- min((cum - cummax(cum)) / cummax(cum))
  
  pct_no_trade <- if ("n_pairs_post" %in% names(df_rets))
    mean(df_rets$n_pairs_post == 0) * 100 else
      mean(rets == 0) * 100
  avg_stocks_kept     <- if ("n_stocks_kept"     %in% names(df_rets))
    mean(df_rets$n_stocks_kept,     na.rm = TRUE) else NA
  avg_stocks_filtered <- if ("n_stocks_filtered" %in% names(df_rets))
    mean(df_rets$n_stocks_filtered, na.rm = TRUE) else NA
  
  data.frame(
    config              = label,
    mean_ret_pct        = round(mean_ret_y * 100, 4),
    total_ret_pct       = round(total_ret  * 100, 2),
    sharpe              = round(sharpe,            4),
    max_dd_pct          = round(max_dd     * 100,  2),
    pct_no_trade        = round(pct_no_trade,       1),
    avg_stocks_kept     = round(avg_stocks_kept,    1),
    avg_stocks_filtered = round(avg_stocks_filtered,1),
    n_months            = length(rets_clean)
  )
}

###########################################################################
# RUN: tutte le configurazioni x tutte le soglie
###########################################################################

cat("\n[BLOCCO 6/7] Strategia FKM con filtro tau (10 config × 7 tau × 203 mesi)...\n")
t0_tau <- Sys.time()

all_rets <- list()
all_perf <- list()

for (label in names(refit_results)) {
  for (tau in tau_grid) {
    
    run_label <- paste0(label, "_tau", tau)
    cat("  Running:", run_label, "...\n")
    
    df_rets <- run_strategy_tau(
      config_fits   = refit_results[[label]],
      final_df      = final_df,
      months_sorted = months_sorted,
      tau           = tau
    )
    
    perf <- calc_performance(df_rets, label = run_label)
    
    # FIX: gestione NULL se tau troppo alto
    if (is.null(perf)) {
      cat("  -> Nessun dato (tau troppo alto o zero trade)\n")
      next
    }
    
    all_rets[[run_label]] <- df_rets
    all_perf[[run_label]] <- perf
    
    cat("  Mean:", perf$mean_ret_pct, "%",
        "| Sharpe:", perf$sharpe,
        "| MaxDD:", perf$max_dd_pct, "%",
        "| Kept:", perf$avg_stocks_kept,
        "| Filtered:", perf$avg_stocks_filtered, "\n")
  }
}

###########################################################################
# BENCHMARK: SHORT-TERM REVERSAL
###########################################################################

cat("\nBenchmark: Short-term Reversal...\n")

reversal_rets <- sapply(seq_along(months_sorted)[-length(months_sorted)],
                        function(t_idx) {
                          
                          t_month  <- months_sorted[t_idx]
                          t1_month <- months_sorted[t_idx + 1]
                          
                          # segnale = ret al mese t (r_t), outcome = ret al mese t+1
                          dfm_t  <- final_df %>% filter(ym == t_month)  %>% select(permno, signal = ret)
                          dfm_t1 <- final_df %>% filter(ym == t1_month) %>% select(permno, ret)
                          
                          dfm <- inner_join(dfm_t, dfm_t1, by = "permno") %>%
                            filter(!is.na(signal), !is.na(ret))
                          
                          if (nrow(dfm) < 10) return(NA)
                          
                          q <- quantile(dfm$signal, probs = c(0.2, 0.8), na.rm = TRUE)
                          mean(dfm$ret[dfm$signal <= q[1]]) - mean(dfm$ret[dfm$signal >= q[2]])
                        })

reversal_df   <- data.frame(
  ym  = months_sorted[-length(months_sorted)],
  ret = reversal_rets
)
reversal_perf <- calc_performance(reversal_df, label = "Reversal_Benchmark")

###########################################################################
# TABELLA FINALE
###########################################################################

perf_final <- bind_rows(c(all_perf, list(reversal_perf)))
print(perf_final)

###########################################################################
# SALVATAGGIO
###########################################################################

saveRDS(all_rets,   file.path(OUT_DIR, "strategy_returns_tau.rds"))
saveRDS(perf_final, file.path(OUT_DIR, "performance_table_tau.rds"))
write.csv(perf_final, file.path(OUT_DIR, "performance_table_tau.csv"), row.names = FALSE)

cat("  ✓ Strategia tau completata in",
    round(difftime(Sys.time(), t0_tau, units = "secs")), "sec\n")

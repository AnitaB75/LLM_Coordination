# Path resolves from this script's own location, so the project can be moved.
.script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) normalizePath(dirname(f)) else normalizePath(getwd())
}

#!/usr/bin/env Rscript
# Visualize how the 4 LLM players choose J / N / M across the 14 rounds of each
# game. Reads the most recent (possibly in-progress) results CSV and counts, per
# round, how many of that game's LLM slots picked each letter. Bots are excluded
# via the per-game `bot_slots` column. Renders with png() + print() (not ggsave).

suppressMessages(library(ggplot2))

dir <- .script_dir()
files <- list.files(dir, pattern = "^name_game_bot_follow_human_.*\\.csv$", full.names = TRUE)
if (length(files) == 0) stop("No results CSV found yet.")
f <- files[order(file.info(files)$mtime, decreasing = TRUE)][1]
d <- read.csv(f, stringsAsFactors = FALSE)
if (nrow(d) == 0) stop("Results CSV is empty (run just started).")

# bot_slots is a 0-indexed list like "[2, 5]"; LLM slots are the other four.
parse_bots <- function(s) as.integer(strsplit(gsub("\\[|\\]|\\s", "", s), ",")[[1]])
pcols <- paste0("p", 1:6)

agg <- do.call(rbind, lapply(seq_len(nrow(d)), function(i) {
  bots <- parse_bots(d$bot_slots[i])          # 0-indexed bot slots
  llm <- setdiff(0:5, bots) + 1               # 1-indexed LLM columns
  ch <- toupper(unlist(d[i, pcols])[llm])
  data.frame(
    game  = d$game[i], round = d$round[i],
    J = sum(ch == "J"), N = sum(ch == "N"),
    M = sum(ch == "M"), Other = sum(!ch %in% c("J", "N", "M"))
  )
}))

long <- reshape(agg, varying = c("J", "N", "M", "Other"), v.names = "count",
                timevar = "letter", times = c("J", "N", "M", "Other"),
                direction = "long")
long$letter <- factor(long$letter, levels = c("J", "N", "M", "Other"))

cols <- c(J = "#1f77b4", N = "#d62728", M = "#2ca02c", Other = "grey70")

p <- ggplot(long, aes(round, count, color = letter)) +
  annotate("rect", xmin = 7.5, xmax = 10.5, ymin = -Inf, ymax = Inf,
           alpha = 0.12, fill = "red") +
  geom_line(linewidth = 1) +
  geom_point(size = 1.7) +
  facet_wrap(~ game, ncol = 1, labeller = label_both) +
  scale_color_manual(values = cols) +
  scale_x_continuous(breaks = 1:14) +
  scale_y_continuous(breaks = 0:4, limits = c(0, 4)) +
  labs(
    title = "LLM player choices over rounds (4 LLMs per game)",
    subtitle = "Red band = rounds 8-10, when the two 'human' bots defect to N",
    x = "Round", y = "# of 4 LLMs choosing letter", color = "Letter"
  ) +
  theme_bw(base_size = 12)

out <- file.path(dir, "llm_jn_choices.png")
png(out, width = 1100, height = 1400, res = 130)
print(p)
dev.off()
cat("Wrote", out, "\n  source:", f, "  rows so far:", nrow(d), "\n")

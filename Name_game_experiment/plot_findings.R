# Path resolves from this script's own location, so the project can be moved.
.script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) normalizePath(dirname(f)) else normalizePath(getwd())
}

#!/usr/bin/env Rscript
# Headline findings figure: across all 5 games, how many of each game's 4 LLM
# players chose J / N / M per round (mean +/- SD across games). Highlights the
# bot-defection window (rounds 8-10, bots play N) and the bot-revert point
# (round 11+, bots return to J) to show the LLMs' lag -> cascade -> overshoot.
# Renders with png() + print() (not ggsave), per project preference.

suppressMessages(library(ggplot2))

dir <- .script_dir()
files <- list.files(dir, pattern = "^name_game_bot_follow_human_.*\\.csv$", full.names = TRUE)
if (length(files) == 0) stop("No results CSV found.")
f <- files[order(file.info(files)$mtime, decreasing = TRUE)][1]
d <- read.csv(f, stringsAsFactors = FALSE)

parse_bots <- function(s) as.integer(strsplit(gsub("\\[|\\]|\\s", "", s), ",")[[1]])
pcols <- paste0("p", 1:6)

# Per game x round: count of the 4 LLM slots choosing each letter.
per <- do.call(rbind, lapply(seq_len(nrow(d)), function(i) {
  bots <- parse_bots(d$bot_slots[i]); llm <- setdiff(0:5, bots) + 1
  ch <- toupper(unlist(d[i, pcols])[llm])
  data.frame(game = d$game[i], round = d$round[i],
             J = sum(ch == "J"), N = sum(ch == "N"), M = sum(ch == "M"))
}))

# Mean +/- SD across games, per round, per letter.
agg <- do.call(rbind, lapply(c("J", "N", "M"), function(L) {
  s <- aggregate(per[[L]], list(round = per$round), function(x) c(m = mean(x), sd = sd(x)))
  data.frame(round = s$round, letter = L, mean = s$x[, "m"], sd = s$x[, "sd"])
}))
agg$letter <- factor(agg$letter, levels = c("J", "N", "M"))
n_games <- length(unique(per$game))

cols <- c(J = "#1f77b4", N = "#d62728", M = "#2ca02c")

p <- ggplot(agg, aes(round, mean, color = letter, fill = letter)) +
  # bot-defection window (bots play N)
  annotate("rect", xmin = 7.5, xmax = 10.5, ymin = -0.15, ymax = 4.15,
           alpha = 0.12, fill = "red") +
  annotate("text", x = 9, y = 4.05, label = "bots play N", size = 3.3, color = "grey25") +
  # bot revert point
  geom_vline(xintercept = 10.5, linetype = "dashed", color = "grey45") +
  annotate("text", x = 12.5, y = 0.30, label = "bots back to J (round 11+)",
           size = 3.3, color = "grey25") +
  geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd), color = NA, alpha = 0.15) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  scale_color_manual(values = cols) +
  scale_fill_manual(values = cols) +
  scale_x_continuous(breaks = 1:14) +
  scale_y_continuous(breaks = 0:4, limits = c(-0.15, 4.15)) +
  labs(
    title = "LLMs follow the two 'humans' to N — with a lag, then overshoot",
    subtitle = sprintf("Mean # of the 4 LLM players choosing each letter per round (±SD across %d games)", n_games),
    x = "Round", y = "# of 4 LLMs choosing letter (mean ± SD)", color = "Letter", fill = "Letter",
    caption = paste(
      "Round 8: 0 LLMs follow in every game (1-round lag).  Rounds 9-10: cascade to N.",
      "Round 11+: LLMs stay on N though the bots reverted to J (overshoot).  M never chosen (equal-value control).",
      sep = "\n")
  ) +
  theme_bw(base_size = 12) +
  theme(plot.caption = element_text(hjust = 0, color = "grey30"))

out <- file.path(dir, "llm_findings_aggregate.png")
png(out, width = 1200, height = 800, res = 130)
print(p)
dev.off()
cat("Wrote", out, "| games:", n_games, "| source:", basename(f), "\n")

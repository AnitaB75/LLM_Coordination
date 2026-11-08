## Academic journal figure style for ggplot2 — the R counterpart of
## ~/.matplotlib/stylelib/academic.mplstyle
##
## Usage:
##   source("theme_academic.R")
##   theme_set(theme_academic())      # applies to every plot in the session
##   print(p)                         # review inline; do not ggsave()
##
## Arial, plain black text, thin black axes, no decorative weight. Distinguish
## series by shape or linetype as well as tone so figures survive greyscale
## printing — the theme cannot enforce that, so set scale_shape_manual() /
## scale_linetype_manual() per plot.

theme_academic <- function(base_size = 9, base_family = "Arial") {
  ggplot2::theme_bw(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      text             = ggplot2::element_text(colour = "black"),
      axis.text        = ggplot2::element_text(colour = "black", size = base_size),
      axis.title       = ggplot2::element_text(colour = "black", size = base_size),
      axis.ticks       = ggplot2::element_line(colour = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(3.5, "pt"),
      axis.line        = ggplot2::element_line(colour = "black", linewidth = 0.3),
      panel.border     = ggplot2::element_blank(),
      panel.grid       = ggplot2::element_blank(),
      panel.background = ggplot2::element_blank(),
      plot.background  = ggplot2::element_blank(),
      # no title/subtitle inside the axes — that text belongs in the caption
      plot.title       = ggplot2::element_blank(),
      plot.subtitle    = ggplot2::element_blank(),
      plot.caption     = ggplot2::element_text(colour = "black", size = base_size - 1,
                                               hjust = 0),
      legend.key       = ggplot2::element_blank(),
      legend.background = ggplot2::element_blank(),
      legend.title     = ggplot2::element_blank(),
      legend.text      = ggplot2::element_text(colour = "black", size = base_size),
      strip.background = ggplot2::element_blank(),
      strip.text       = ggplot2::element_text(colour = "black", size = base_size)
    )
}

## Greyscale-safe defaults for two series, matching the matplotlib figures:
## filled circle vs open square, both with black edges.
scale_shape_academic <- function(...) {
  ggplot2::scale_shape_manual(values = c(16, 22), ...)
}
scale_fill_academic <- function(...) {
  ggplot2::scale_fill_manual(values = c("black", "white"), ...)
}

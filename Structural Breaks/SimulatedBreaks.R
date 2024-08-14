library(changepoint)
library(sarbcurrent)
library(tidyverse)
library(lubridate)
library(quantmod)
library(strucchange)
library(urca)
library(quantmod)
library(WaveletComp)


rm(list=ls())
graphics.off()


# Change in Mean ----------------------------------------------------------


set.seed(42)
sim_mean <- c(rnorm(100, 0, 1),
              rnorm(50, 1.5, 1),
              rnorm(90, 0, 1),
              rnorm(120, -0.8, 1))
png("diff_mean.png",width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_mean,main="Diferencia de Medias")
dev.off()



png("fig1.png",width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_mean)
m_binseg <- cpt.mean(sim_mean, penalty = "BIC", method = "BinSeg", Q = 5)
plot(m_binseg, type = "l", xlab = "Index", cpt.width = 4,main="BinSeg")
dev.off()

png("fig2.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_mean)
m_segneigh <- cpt.mean(sim_mean, penalty = "BIC", method = "SegNeigh", Q = 5)
plot(m_segneigh, type = "l", xlab = "Index", cpt.width = 4,,main="SegNeigh")
dev.off()
cpts(m_segneigh)

m_pelt <- cpt.mean(sim_mean, penalty = "BIC", method = "PELT")
png("fig3.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_mean,main="PELT")
plot(m_pelt, type = "l", cpt.col = "blue", 
     xlab = "Index", cpt.width = 4,,main="PELT")
cpts(m_pelt)
dev.off()


m_pm <- cpt.mean(sim_mean, penalty = "Manual", pen.value = "1.5 * log(n)",
                 method = "PELT")
png("fig4.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_mean)
plot(m_pm, type = "l", cpt.col = "red", xlab = "Index", 
     cpt.width = 4,main="1.5 * log(n)")
dev.off()


# Simulated Data - Change in Variance -------------------------------------

sim_var <- c(rnorm(100, 0, 1),
             rnorm(50, 0, 2),
             rnorm(90, 0, 1),
             rnorm(120, 0, 0.5))

png("diff_var.png",width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_var,main="Diferencia en Varianzas")
dev.off()


png("fig5.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_var)
v_pelt <- cpt.var(sim_var, method = "PELT")
plot(v_pelt, type = "l", cpt.col = "blue", 
     xlab = "Index", cpt.width = 4, main="PELT")
dev.off()

cpts(v_pelt)


# Change in Mean and Variance ---------------------------------------------

sim_mv <- c(rnorm(100, 0, 1),
            rnorm(50, 1, 2),
            rnorm(90, 0, 1),
            rnorm(120, -0.8, 0.5))

png("diff_mean_var.png",width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_mv,main="Diferencia en Media y Varianzas")
dev.off()


png("fig6.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(sim_mv)
mv_pelt <- cpt.meanvar(sim_mv, method = "PELT")
plot(mv_pelt,main="PELT")
dev.off()



# Simulated Data ----------------------------------------------------------

set.seed(42)

x1 <- arima.sim(model = list(ar = 0.9), n = 100)
x2 <- arima.sim(model = list(ma = 0.1), n = 100)
x3 <- arima.sim(model = list(ar = 0.5, ma = 0.3), n = 100)

y <- c((1 + x1),
       x2,
       (0.5 - x3))

png("diff_arima.png",width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(y,main="Tres Arimas Diferentes")
dev.off()



dat <- tibble(ylag0 = y,
              ylag1 = lag(y)
) %>%
  drop_na()

qlr <- Fstats(ylag0 ~ ylag1, data = dat)

breakpoints(qlr)
sctest(qlr, type = "supF")
png("fig7.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot(qlr,main="Lag0 ~ Lag1")
dev.off()

cusum <- efp(ylag0 ~ ylag1, type = "OLS-CUSUM", data = dat)
png("fig8.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot(cusum)
dev.off()

png("fig9.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
plot.ts(y)
mv_pelt <- cpt.meanvar(y, method = "PELT")
plot(mv_pelt,main="PELT Media/Varianza")
dev.off()


simulated_df <- data.frame(date = 1:length(y), price = y)

simulated_wavelet <- analyze.wavelet(simulated_df, my.series = "price", loess.span = 0)

png("fig10.png", width = 1024, height = 768, units = "px",res = 300,
    pointsize = 8)
wt.image(simulated_wavelet, main = "Espectro de Potencia de los Datos Simulados")
dev.off()

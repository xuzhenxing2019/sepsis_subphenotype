#install.packages("survminer")
# Loading
library("survminer")
# Fit survival curves
require("survival")

#loading data
SepsisData<-readr::read_csv(paste0("/Users/xuzhenxing/Documents/sepsis/","survival_analysis_df.csv"))

fit<- survfit(Surv(hosp_los, hospital_expire_flag) ~ group, data = SepsisData)
summary(fit)

# Drawing survival curves
surp<-ggsurvplot(fit,
                 legend.title = "Subphenotypes",
                 legend.labs = c("DI","RI","DW","RW"),
                 #legend.labs = c("cluster_0", "cluster_1","cluster_2","cluster_3"),
                 palette = c("purple3","darkgreen","blue","red"),
                 #size = 0.5,
                 censor = FALSE, # True

                 # font.main = 20,
                 # font.x = 15,
                 # font.y = 15,
                 # font.tickslab = 15,
                 conf.int = TRUE, # Add confidence interval
                 pval = TRUE, # Add p-value
                 #pval.method = TRUE,
                 surv.plot.height = 0.5,
                 
                 risk.table = TRUE,        # Add risk table
                 risk.table.col = "strata",# Risk table color by groups
                 risk.table.height = 0.25, # Useful to change when you have multiple groups
                 fontsize = 8.5,  # 5,9
                 #risk.table.fontsize = 7,
                 
                 xlab = 'Time (Day)',
                 #xscale = 30,
                 #xscale = "d_m",
                 xlim = c(0, 27),
                 ylim = c(0.5, 1),
                 break.time.by = 1,
                 ggtheme = theme_bw()
                 
)

ggpar_plot<-ggpar(
  surp,
  font.title    = c(30),  
  font.x        = c(30),          
  font.y        = c(30),
  font.tickslab = 30, # 16
  font.legend = list(size = 30)
)
ggpar_plot

# save survival figure
ggsave(paste0("/Users/xuzhenxing/Documents/Sepsis/sepsis_1225/data/survival_3_26_MIMIC/","MIMIC_survival_analysis_NumCluster_4_risk_table.pdf"), surp$plot,width = 15, height = 30)




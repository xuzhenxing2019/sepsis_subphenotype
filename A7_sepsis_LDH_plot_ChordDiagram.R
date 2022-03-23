#install.packages("circlize")
library(circlize)
library(readr)

# plot Chord Diagram among all clusters
figure_title_s <- c("Chord_data_df_combined","Chord_data_df_cluster_0","Chord_data_df_cluster_1","Chord_data_df_cluster_2","Chord_data_df_cluster_3") # c("cluster_0","cluster_1","cluster_2","cluster_3") -> c("DI","RI","DW","RW"),

for (i in figure_title_s) {
  #print("starting :",i)
  figure_title = i
  # starting reading data for eICU
  SepsisData<-read.csv(paste0("/Users/xuzhenxing/Documents/eICU_AKI_Sepsis/ChordDiagrams/comorbidity/",figure_title,".csv"),row.names= 1)

# plot ChordDiagram in terms of Inflammatory,Hepatic,Cardiovascular,Renal,Hematologic,Pulmonary,Neurologic,Comorbidity
  
  # for all clusters
  if (figure_title == "Chord_data_df_combined"){
    grid_col <-  c(DI = "purple3", RI = "darkgreen", DW = "blue", RW = "red",
                   Inflammatory = "grey", Hepatic = "grey", Cardiovascular = "grey", Renal = "grey", Hematologic = "grey", Pulmonary = "grey", Neurologic = "grey", Comorbidity = "grey")
  }
  # for cluster 0
  if (figure_title == "Chord_data_df_cluster_0"){
    grid_col <-  c(DI = "purple3", RI = "grey", DW = "grey", RW = "grey",
                   Inflammatory = "grey", Hepatic = "grey", Cardiovascular = "grey", Renal = "grey", Hematologic = "grey", Pulmonary = "grey", Neurologic = "grey", Comorbidity = "grey")
  }
  
  # for cluster 1
  if (figure_title == "Chord_data_df_cluster_1"){
    grid_col <-  c(DI = "grey", RI = "darkgreen", DW = "grey", RW = "grey",
                   Inflammatory = "grey", Hepatic = "grey", Cardiovascular = "grey", Renal = "grey", Hematologic = "grey", Pulmonary = "grey", Neurologic = "grey", Comorbidity = "grey")
  }
  
  # for cluster 2
  if (figure_title == "Chord_data_df_cluster_2"){
    grid_col <-  c(DI = "grey", RI = "grey", DW = "blue", RW = "grey",
                   Inflammatory = "grey", Hepatic = "grey", Cardiovascular = "grey", Renal = "grey", Hematologic = "grey", Pulmonary = "grey", Neurologic = "grey", Comorbidity = "grey")
  }
  
  # for cluster 3
  if (figure_title == "Chord_data_df_cluster_3"){
    grid_col <-  c(DI = "grey", RI = "grey", DW = "grey", RW = "red",
                   Inflammatory = "grey", Hepatic = "grey", Cardiovascular = "grey", Renal = "grey", Hematologic = "grey", Pulmonary = "grey", Neurologic = "grey", Comorbidity = "grey")
  }
  

# plot ChordDiagram in terms of six subscores Respiration, Coagulation, Liver, Cardiovascular, CNS, Renal

  # # for all clusters
  # if (figure_title == "Chord_data_df_combined"){
  #   grid_col <-  c(cluster_0 = "purple3", cluster_1 = "darkgreen", cluster_2 = "blue", cluster_3 = "red",
  #                  Respiration = "grey", Coagulation = "grey", Liver = "grey", Cardiovascular = "grey", CNS = "grey", Renal = "grey")
  # }
  # 
  # # for cluster 0
  # if (figure_title == "Chord_data_df_cluster_0"){
  #   grid_col <-  c(cluster_0 = "purple3", cluster_1 = "grey", cluster_2 = "grey", cluster_3 = "grey",
  #                  Respiration = "grey", Coagulation = "grey", Liver = "grey", Cardiovascular = "grey", CNS = "grey", Renal = "grey")
  # }
  # 
  # # for cluster 1
  # if (figure_title == "Chord_data_df_cluster_1"){
  #   grid_col <-  c(cluster_0 = "grey", cluster_1 = "darkgreen", cluster_2 = "grey", cluster_3 = "grey",
  #                  Respiration = "grey", Coagulation = "grey", Liver = "grey", Cardiovascular = "grey", CNS = "grey", Renal = "grey")
  # }
  # 
  # # for cluster 2
  # if (figure_title == "Chord_data_df_cluster_2"){
  #   grid_col <-  c(cluster_0 = "grey", cluster_1 = "grey", cluster_2 = "blue", cluster_3 = "grey",
  #                  Respiration = "grey", Coagulation = "grey", Liver = "grey", Cardiovascular = "grey", CNS = "grey", Renal = "grey")
  # }
  # 
  # #  for cluster 3
  # if (figure_title == "Chord_data_df_cluster_3"){
  #   grid_col <-  c(cluster_0 = "grey", cluster_1 = "grey", cluster_2 = "grey", cluster_3 = "red",
  #                  Respiration = "grey", Coagulation = "grey", Liver = "grey", Cardiovascular = "grey", CNS = "grey", Renal = "grey")
  # }
  
  circos.par(gap.after = c(rep(2, nrow(mat)-1), 10, rep(2, ncol(mat)-1), 10))
  chord_figure = chordDiagram(mat, annotationTrack = "grid", transparency = 0.5, grid.col = grid_col,
                              preAllocateTracks = list(track.height = 0.03),annotationTrackHeight = c(0.03))
  #for(si in get.all.sector.index()) {circos.axis(h = "top", labels.cex = 0.3, sector.index = si, track.index = 2)}
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    xlim = get.cell.meta.data("xlim")
    ylim = get.cell.meta.data("ylim")
    sector.name = get.cell.meta.data("sector.index")
    
    #circos.lines(xlim, c(mean(ylim), mean(ylim)), lty = 3)
    #for(p in seq(0, 1, by = 0.25)) {circos.text(p*(xlim[2] - xlim[1]) + xlim[1], mean(ylim), p, cex = 0.4, adj = c(0.5, -0.2), niceFacing = TRUE)}
    circos.text(mean(xlim), 1.4, sector.name, niceFacing = TRUE, cex = 1.75)
  }, bg.border = NA)
  
  
  circos.clear()
  dev.off()
  
}




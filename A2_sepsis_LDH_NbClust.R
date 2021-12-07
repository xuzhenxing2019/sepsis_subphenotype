# NbCluster for obtaining the best number of clusters

#install.packages("NbClust")
library("NbClust")

set.seed(1)
# x<-rbind(matrix(rnorm(150,sd=0.3),ncol=3),
#          matrix(rnorm(150,mean=3,sd=0.2),ncol=3),
#          matrix(rnorm(150,mean=5,sd=0.3),ncol=3))
# 
# diss_matrix<- dist(x, method = "euclidean", diag=FALSE)


# load data
#x<-readr::read_csv(paste0("/Users/xuzhenxing/Documents/Sepsis/sepsis_1225/NbClust/","data.csv"))
#diss_matrix <- read.csv("/Users/xuzhenxing/Documents/Sepsis/sepsis_1225/NbClust/dis_matrix.csv", header=FALSE)
#diss_matrix<-as.dist(diss_matrix)

x<-readr::read_csv(paste0("/Users/xuzhenxing/Documents/Sepsis/sepsis_1225/NbClust/","CEDAR_data.csv"))
diss_matrix <- read.csv("/Users/xuzhenxing/Documents/Sepsis/sepsis_1225/NbClust/CEDAR_dis_matrix.csv", header=FALSE)
diss_matrix<-as.dist(diss_matrix)

index_s <- c("mcclain") # "gap", "duda","pseudot2", "rubin", "kl", "ch", "hartigan", "ccc", "scott", "marriot", "trcovw", "tracew", "friedman",
# "cindex", "db", "silhouette","beale", "ratkowsky", "ball", "ptbiserial",
# "frey", "mcclain", "dunn", "sdindex", "sdbw"
# ) #   

method_s <- c( "complete") # 3,4,5,6,7,8, # "ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid", "kmeans"
distance_s<- c("euclidean") # ,"manhattan", "maximum", "canberra", "binary", "minkowski" 

for (k_index in index_s) {
  cat("index----------------------------------------- :",k_index,"\n")
  for (i in method_s) {
    # print("method--- :")
    # print(i)
    cat("method----------------------------------------- :",i,"\n")
    method = i
    # res<-NbClust(x, diss=diss_matrix, distance = NULL, min.nc=2, max.nc=6, method = method, index = k_index)   
    # print(res$Best.nc)
    for (j in distance_s) {
      cat("distance----------------------------------------- :",j,"\n")
      res<-NbClust(x, diss=NULL, distance = j, min.nc=2, max.nc=6, method = method, index = k_index)   
      # res<-NbClust(x, diss=diss_matrix, distance = NULL, min.nc=2, max.nc=6, method = method, index = k_index) 
      #res$All.index
      print(res$Best.nc)
      #res$Best.partition
    }
  }
  
}



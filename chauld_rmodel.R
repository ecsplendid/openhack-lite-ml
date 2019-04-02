  #TODO
  # 1. FirstSeenDate for various OS Versions etc
  # 2. 'Age' derived from FirstSeenDate
  # 3. Variance in RAM/Storage by OEMModelIdentifier
  # 4. Where locale does not match country
  # 5. Encode CensusOSVersion and AVSigVersion to appropriately deal with likely unseen data in test; Lots of feature ideas here https://www.kaggle.com/tunguz/ms-malware-adversarial-validation
  
  #Parallel backend
  library(doParallel)
  library(doMC)
  library(foreach)
  clusterCores <- 20
  library(devtools)
  library(dplyr)
  library(lightgbm)
  library(data.table)
  library(featuretoolsR)
  library(DescTools)
  library(fst)

  set.seed(2001)
  
  #---------------------------
  #TODO:Move loading across to the fst package per https://blog.revolutionanalytics.com/2017/02/fst-fast-serialization-of-r-data-frames.html
  cat("Loading data...\n")
  dt <- fread("../input/train.csv", drop = "MachineIdentifier")
  
  N <- dt[, .N]
  y <- dt[, HasDetections]
  dt[, HasDetections := NULL]
  
  dt <- rbindlist(list(dt,fread("../input/test.csv", drop = "MachineIdentifier")), fill=T,use.names=T,idcol=T)
  
  dt[, .rowId := .I]
  
  dt[,.noise:=rnorm(.N)]

  setkey(dt,.rowId)
  
  #---------------------------
  cat("Adding features...\n")
  
  cat("Ordering version columns") #Per https://www.kaggle.com/sionek/ordering-version-variables
  versionVars <- c("AvSigVersion","EngineVersion","AppVersion","Census_OSVersion")
  # patterns for proper ordering version variables
  pattern1 <- "^([0-9]+)\\D([0-9]+)\\D([0-9]+)\\D([0-9]+)$"
  pattern2 <- "0\\1.0000\\2.0000\\3.0000\\4"
  pattern3 <- "[0-9]*([0-9]{2})\\D[0-9]*([0-9]{5})\\D[0-9]*([0-9]{5})\\D[0-9]*([0-9]{5})"
  pattern4  <- "\\1.\\2.\\3.\\4"
  
  cat("Transforming version variables into ordered integers...\n")
  registerDoMC(length(versionVars))
  cat(paste('Running in parallel with ', getDoParWorkers(), ' workers'))
  
  combinedDT <- foreach (i=1:length(versionVars), .combine=cbind) %dopar% 
  {
    f <- versionVars[i]
    dt[,as.integer(as.factor(gsub(pattern3,pattern4,gsub(pattern1,pattern2,.SD[[f]]))))]
  }
  combinedDT <- setnames(as.data.table(combinedDT),versionVars)
  dt <- cbind(dt[,-(versionVars),with=F],combinedDT)
  combinedDT <- NULL
  registerDoSEQ()
  
  
  dt[, count.AvSigVersion.Wdft_IsGamer := .N / nrow(dt), by = "AvSigVersion,Wdft_IsGamer"
     ][, count.Census_ProcessorCoreCount.Wdft_RegionIdentifier := .N / nrow(dt), by = "Census_ProcessorCoreCount,Wdft_RegionIdentifier"
       ][, count.Census_ProcessorCoreCount.Census_OEMNameIdentifier := .N / nrow(dt), by = "Census_ProcessorCoreCount,Census_OEMNameIdentifier"
         ][, count.GeoNameIdentifier.Census_OEMNameIdentifier.Census_OSBuildRevision := .N / nrow(dt), by = "GeoNameIdentifier,Census_OEMNameIdentifier,Census_OSBuildRevision"
           ][, count.OsBuildLab := .N / nrow(dt), by = "OsBuildLab"
             ]
  
  cat("Fixing SmartScreen Flags")
  dt[,SmartScreenIsValid:=SmartScreen %in% c('','RequireAdmin','ExistsNotSet','Off','Warn','Prompt','Block')]
  
  #---------------------------
  cat("Converting character columns...\n")
  #TODO: Correct handling of categoricals
  cats <- names(which(sapply(dt, is.character)))
  dt[, (cats) := lapply(.SD, function(x) as.integer(as.factor(x))), .SDcols = cats]  
  rm(cats); invisible(gc())  

  #-----------------------------------
  #cat('Deep Feature Synthesis')
  # Create entityset
  #es <- as_entityset(dt, index = "key", entity_id = "dt", id = "entities")
  #es$normalize_entity(base_entity_id = "dt",new_entity_id = "CountryIdentifier",index="CountryIdentifier",make_time_index=F)
  
  #ft_matrix <- 
  #  dfs(es,
  #    target_entity = "dt", 
  #    agg_primitives = as.list("mean")
  #  )
  
  #-----------------------------
  cat('Hand rolled features')
  
  # Variance and moments of various properties by OEMModelIdentifier
  #TODO: Check on NAs
  setkey(dt,Census_OEMModelIdentifier) 
  dt[,CoV.Census_TotalPhysicalRAM:=CoefVar(Census_TotalPhysicalRAM,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for machines that are off the shelf vs home built
  dt[,Skew.Census_TotalPhysicalRAM:=Skew(Census_TotalPhysicalRAM,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for machines that are off the shelf vs home built
  dt[,Kurt.Census_TotalPhysicalRAM:=Kurt(Census_TotalPhysicalRAM,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for machines that are off the shelf vs home built
  
  dt[,CoV.Census_SystemVolumeTotalCapacity:=CoefVar(Census_SystemVolumeTotalCapacity,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for machines that are off the shelf vs home built
  dt[,Skew.Census_SystemVolumeTotalCapacity:=Skew(Census_SystemVolumeTotalCapacity,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for machines that are off the shelf vs home built
  dt[,Kurt.Census_SystemVolumeTotalCapacity:=Kurt(Census_SystemVolumeTotalCapacity,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for machines that are off the shelf vs home built
  
  dt[,CoV.Census_ProcessorCoreCount:=CoefVar(Census_ProcessorCoreCount,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for machines that are off the shelf vs home built
  dt[,Skew.Census_ProcessorCoreCount:=Skew(Census_ProcessorCoreCount,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier]
  dt[,Kurt.Census_ProcessorCoreCount:=Kurt(Census_ProcessorCoreCount,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier]
  
  dt[,CoV.Census_InternalBatteryNumberOfCharges:=CoefVar(Census_InternalBatteryNumberOfCharges,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] #Proxy for duration on market
  dt[,Skew.Census_InternalBatteryNumberOfCharges:=Skew(Census_InternalBatteryNumberOfCharges,na.rm=T),by=Census_OEMModelIdentifier] #Hopefully pulls some measure of the lifecycle of devices
  dt[,Kurt.Census_InternalBatteryNumberOfCharges:=Kurt(Census_InternalBatteryNumberOfCharges,na.rm=T),by=Census_OEMModelIdentifier] #As above
  
  dt[,Mean.AvSigVersion_by_Census_OEMModelIdentifier:=Mean(AvSigVersion,trim=0.1,na.rm=T),by=Census_OEMModelIdentifier] #Is this a wildwest machine type or not?
  dt[,CoV.AvSigVersion_by_Census_OEMModelIdentifier:=CoefVar(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] 
  dt[,Skew.AvSigVersion_by_Census_OEMModelIdentifier:=Skew(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] 
  dt[,Kurt.AvSigVersion_by_Census_OEMModelIdentifier:=Kurt(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=Census_OEMModelIdentifier] 
  
  dt[,CountUnique.AvSigVersion:=n_distinct(AvSigVersion,na.rm=T),by=Census_OEMModelIdentifier]
  
  #Various moments on Country
  setkey(dt,CountryIdentifier)
  dt[,Mean.AvSigVersion_by_CountryIdentifier:=Mean(AvSigVersion,trim=0.1,na.rm=T),by=CountryIdentifier]
  dt[,CoV.AvSigVersion_by_CountryIdentifier:=CoefVar(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] #How good is a country at keeping up to date
  dt[,Skew.AvSigVersion_by_CountryIdentifier:=Skew(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] 
  dt[,Kurt.AvSigVersion_by_CountryIdentifier:=Kurt(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] 
  
  dt[,Mean.AppVersion:=Mean(AppVersion,trim=0.1,na.rm=T),by=CountryIdentifier]
  dt[,CoV.AppVersion:=CoefVar(AppVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] #How good is a country at keeping up to date
  dt[,Skew.AppVersion:=Skew(AppVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] 
  dt[,Kurt.AppVersion:=Kurt(AppVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] 
  
  dt[,Mean.Census_OSVersion:=Mean(Census_OSVersion,trim=0.1,na.rm=T),by=CountryIdentifier]
  dt[,CoV.Census_OSVersion:=CoefVar(Census_OSVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] #How good is a country at keeping up to date
  dt[,Skew.Census_OSVersion:=Skew(Census_OSVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] 
  dt[,Kurt.Census_OSVersion:=Kurt(Census_OSVersion,unbiased=F,conf.level=NA,na.rm=T),by=CountryIdentifier] 
  
  #Various moments on Country
  setkey(dt,Wdft_RegionIdentifier)
  dt[,Mean.AvSigVersion_by_Wdft_RegionIdentifier:=Mean(AvSigVersion,trim=0.1,na.rm=T),by=Wdft_RegionIdentifier]
  dt[,CoV.AvSigVersion_by_Wdft_RegionIdentifier:=CoefVar(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=Wdft_RegionIdentifier] #How good is a country at keeping up to date
  dt[,Skew.AvSigVersion_by_Wdft_RegionIdentifier:=Skew(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=Wdft_RegionIdentifier] 
  dt[,Kurt.AvSigVersion_by_Wdft_RegionIdentifier:=Kurt(AvSigVersion,unbiased=F,conf.level=NA,na.rm=T),by=Wdft_RegionIdentifier] 
  
  #Proportion of Locale by Country. #Dplyr syntax nicer per https://stackoverflow.com/questions/30944116/r-data-table-subgroup-weighted-percent-of-group
  #TODO: Performance improvement via keying the merge
  #TODO: Make nicers with data.table
  #TODO: Build a generalized function for doing frequencies by group!
  library(dplyr)
  freqTable <- as.data.table(dt %>% 
    group_by(CountryIdentifier, LocaleEnglishNameIdentifier) %>%
    summarise(count.LocaleEnglishNameIdentifier.CountryIdentifier = n()) %>%
    mutate(freq.LocaleEnglishNameIdentifier.CountryIdentifier = count.LocaleEnglishNameIdentifier.CountryIdentifier/sum(count.LocaleEnglishNameIdentifier.CountryIdentifier)))
  
  setkeyv(dt, c("CountryIdentifier","LocaleEnglishNameIdentifier"))
  dt <- merge(dt,freqTable,by=c("CountryIdentifier","LocaleEnglishNameIdentifier"))
  

  
#---------------------------
cat("Preparing data for boosting...\n")

setkey(dt,.rowId)
dt[,.rowId:=NULL]

cat("Preparing data for boosting...\n")
dm <- data.matrix(dt)
tr <- lgb.Dataset(data = dm[1:N, ], label = y)
te <- dm[-(1:N), ]

#---------------------------
cat("Training model and predicting...\n")

subm <- fread("../input/sample_submission.csv")


#TODO:a
# Treat NA correctly
# Treat categorical correctly
i <- 0.7
p <- list(boosting_type = "gbdt", 
          objective = "binary",
          metric = "auc", 
          nthread = 26, 
          learning_rate = 0.1, 
          max_depth = 6,
          num_leaves = 40,
          feature_fraction = i, 
          bagging_fraction = i, 
          bagging_freq = 1,
          lambda_l1 = 0.1, 
          lambda_l2 = 0.1,
          device = 'gpu',
          max_bin=63,
          metrics='auc',
          early_stopping_rounds=20)
m_cv <- lgb.cv(p, tr, 10000,nfold=5, verbose=1)

cv_score <- m_cv$best_score
best_iter <- m_cv$best_iter



rm(m_cv); invisible(gc())

#TODO: Parallelize with doAzureParallel
#TODO: Exclude .noise
n_bags <- 3
subm[, HasDetections := 0]
for (i in seq(0.5, 0.9, length.out = n_bags)) {
  p <- list(boosting_type = "gbdt", 
            objective = "binary",
            metric = "auc", 
            nthread = 26, 
            learning_rate = 0.1, 
            max_depth = 6,
            num_leaves = 40,
            sub_feature = i, 
            sub_row = i, 
            bagging_freq = 1,
            lambda_l1 = 0.1, 
            lambda_l2 = 0.1,
            device = 'gpu',
            max_bin=63,
            metrics='auc')
  
  m_gbm <- lgb.train(p, tr, best_iter*1.1,  verbose = 0)
  
  
  subm[, HasDetections := HasDetections + predict(m_gbm, te) / n_bags]
  
  #rm(m_gbm); invisible(gc())
}

#Feature importance
featImp <- lgb.importance(m_gbm)
lgb.plot.importance(featImp,top_n = 50,left_margin = 15)

#Sanity check
hist(subm$HasDetections)


#---------------------------
cat("Making submission file...\n")

fwrite(subm, "ms_malware_new.csv")

# ***** Import necessary libraries *****
library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)

# ***** Import dataset *****
df <- fread("mushrooms.csv")
df %>% skim()
df %>% View()



func <- function(df){
  gsub("\\'","", df)
  gsub("\\'" ,"", df)
}
df <- apply(df, 2, func) %>% as.data.frame()

# ****** Set target as factor *****
df$class %>% table() %>% prop.table()
df$class <- df$class %>% recode(" 'e' =1 ; 'p' = 0") %>% as.factor()
class(df$class)

names(df) <- names(df) %>%
  str_replace_all(" ","_") %>%
  str_replace_all("-","_") %>%
  str_replace_all("\\(","_") %>%
  str_replace_all("\\)","") %>%
  str_replace_all("\\/","_") %>%
  str_replace_all("\\:","_") %>%
  str_replace_all("\\.","_") %>%
  str_replace_all("\\,","_") %>%
  str_replace_all("\\%","_")


colnames(df)
# ** Change to factor according to issues in dataset **
df$cap_shape <- df$cap_shape %>% as.factor()
df$cap_surface <- df$cap_surface %>% as.factor()
df$cap_color <- df$cap_color %>% as.factor()
df$bruises_3F <- df$bruises_3F %>% as.factor()
df$odor <- df$odor %>% as.factor()
df$gill_attachment <- df$gill_attachment %>% as.factor()
df$gill_spacing <- df$gill_spacing %>% as.factor()
df$gill_size <- df$gill_size %>% as.factor()
df$gill_color <- df$gill_color %>% as.factor()
df$stalk_shape <- df$stalk_shape %>% as.factor()
df$stalk_root <- df$stalk_root %>% as.factor()
df$stalk_color_above_ring <- df$stalk_color_above_ring %>% as.factor()
df$stalk_color_below_ring <- df$stalk_color_below_ring %>% as.factor()
df$stalk_surface_above_ring <- df$stalk_surface_above_ring %>% as.factor()
df$stalk_surface_below_ring<- df$stalk_surface_below_ring %>% as.factor()
df$veil_type <- df$veil_type %>% as.factor()
df$veil_color <- df$veil_color %>% as.factor()
df$ring_number <- df$ring_number %>% as.factor()
df$ring_type <- df$ring_type %>% as.factor()
df$spore_print_color <- data$spore_print_color %>% as.factor()
df$population <- df$population %>% as.factor()
df$habitat <- df$habitat %>% as.factor()

# ********* Modelling ********
h2o.init()
h2o_data <- df %>% as.h2o()

# ********** Splitting data into train and test ******
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- df %>% select(-class) %>% names()

# ******************* Fitting h2o model ***************
model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  nfolds = 10, seed = 123,
  max_runtime_secs = 360)

model@leaderboard %>% as.data.frame()
model@leader 

# ************ Predicting the Test set results *********
pred <- model@leader %>% h2o.predict(test) %>% as.data.frame()

# *************** Threshold / Cutoff **********
model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1') -> treshold

# *************** Model evaluation ************
# *************** Confusion Matrix ************

model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.metric() %>% 
  select(threshold,precision,recall,tpr,fpr) %>% 
  add_column(tpr_r=runif(nrow(.),min=0.001,max=1)) %>% 
  mutate(fpr_r=tpr_r) %>% 
  arrange(tpr_r,fpr_r) -> deep_metrics

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc() %>% round(2) -> auc

highchart() %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y=tpr,x=fpr), color='green', name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y=tpr_r,x=fpr_r), color='red', name='Random Guess') %>% 
  hc_add_annotation(
    labels = list(
      point = list(xAxis=0,yAxis=0,x=0.3,y=0.6),
      text = glue('AUC = {enexpr(auc)}'))
  ) %>%
  hc_title(text = "ROC Curve") %>% 
  hc_subtitle(text = "Model is performing much better than random guessing") 

# ******* Check overfitting **********

model@leader %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)

# ***** Save model ****

model@leaderboard %>% as_tibble() %>% slice(1) %>% pull(model_id) %>% 
  h2o.getModel() %>% h2o.saveModel(path = path)

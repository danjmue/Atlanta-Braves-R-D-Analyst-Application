library(tidyverse)
library(cluster)
library(factoextra)
library(randomForest)

# Loading in the data
read.csv('PitchData/PitchData.csv') -> data

# Grabbing only pitches that were curveballs
data %>%
  filter(Pitch_Type == 'Curveball') -> curveballs

# Taking the absolute value of x movement and release position to eliminate handedness
curveballs %>%
  mutate(
    x_movement = abs(x_movement),
    release_pos_x = abs(release_pos_x)
  ) -> curveballs

# Creating binomial variable Whiff to label a pitch as a swing and miss
curveballs %>%
  mutate(
    whiff = ifelse(Pitch_Outcome == 'StrikeSwinging', 1, 0)
  ) -> curveballs

# Standardizing pitch metrics
curveballs %>%
  mutate_at(c('release_speed', 'x_movement', 'z_movement', 'release_spin_rate', 'spin_dir',
              'release_pos_x', 'release_pos_z', 'release_extension', 'plate_x', 'plate_z'), function(x) {
                (x - mean(x, na.rm = T)) / sd(x, na.rm = T)
              }) %>%
  na.omit() -> curveballs_standardized

### Logistic Regression Model

# Grabbing the standardized pitch metrics from the dataframe to cluster them
curveballs_standardized[,15:24] -> curveballs_clustered

# Creating kmeans clusters with three centers
k3 <- kmeans(curveballs_clustered, centers = 3, nstart = 25)

# Plotting the clusters
fviz_cluster(k3, geom = "point", data = curveballs_clustered) + ggtitle("k = 3")

# Adding the clusters to the original dataframe
curveballs_standardized %>%
  mutate(
    k3_cluster = k3$cluster,
  ) -> curveballs_standardized

# Creating data frame of coefficients for the plot
coeffs <- data.frame(matrix(0, 3, 13))

# Running logistic regressions on each cluster and printing their summaries, confusion matrices, and test accuracy scores
lapply(1:3, function(i) {
  set.seed(1)
  curveballs_standardized %>%
    filter(k3_cluster == i) -> df
  
  inds <- sample(1:nrow(df), .7 * nrow(df))
  
  df[inds,] -> training
  df[-inds,] -> testing
  
  glm_fit = glm(whiff ~ 
                  Balls +
                  Strikes +
                  release_speed + 
                  x_movement + 
                  z_movement + 
                  release_spin_rate + 
                  spin_dir + 
                  release_pos_x + 
                  release_pos_z + 
                  release_extension + 
                  plate_x + 
                  plate_z, training, family = 'binomial')
  summary(glm_fit) %>%
    print()
  
  testing$whiff_pred <- predict(glm_fit, testing, type = 'response')
  
  testing %>%
    mutate(whiff_pred_binary = ifelse(whiff_pred > .5, 1, 0)) -> testing
  
  table(testing$whiff, testing$whiff_pred_binary) %>%
    print()
  
  coeffs[i,] <<- glm_fit$coeff
  names(coeffs) <<- names(glm_fit$coeff)
  
  return(mean(testing$whiff == testing$whiff_pred_binary))
})

# Generating coefficient plot for logistic regression models
coeffs$cluster <- 1:3
coeffs %>%
  pivot_longer(-cluster, names_to = 'Absolute Value of Parameter', values_to = 'Coefficient') %>%
  mutate(Coefficient = abs(Coefficient)) %>%
  ggplot(aes(`Absolute Value of Parameter`, Coefficient, color = factor(cluster))) +
  geom_point(size = 3) +
  labs(title = 'Coefficient Magnitute for each Cluster\'s Parameters',
       color = 'Cluster')
ggsave('plot.png', width = 5000, height = 2500, units = 'px')

### Random Forest model

# Creating a train/test split for the random forest model
set.seed(1)
inds <- sample(1:nrow(curveballs_standardized), .7 * nrow(curveballs_standardized))
curveballs_standardized[inds,] -> training
curveballs_standardized[-inds,] -> testing

# Building random forest
rf <- randomForest(whiff ~ 
               Balls +
               Strikes +
               release_speed + 
               x_movement + 
               z_movement + 
               release_spin_rate + 
               spin_dir + 
               release_pos_x + 
               release_pos_z + 
               release_extension + 
               plate_x + 
               plate_z, training)

# Printing random forest
rf

# Plotting importance of each parameter in the random forest
varImpPlot(rf)

# Making predictions on the testing set
predict(rf, testing) -> testing$whiff_pred
testing %>%
  mutate(whiff_pred_binary = ifelse(whiff_pred > .5, 1, 0)) -> testing

# Confusion matrix
table(testing$whiff, testing$whiff_pred_binary) %>%
  print()

# Test accuracy
print(mean(testing$whiff == testing$whiff_pred_binary))

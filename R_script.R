# "Capstone Edx Project" by Peter Park "2026-06-02"

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(dplyr)
library(ggplot2)
library(dslabs)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#Basic information on the dataset 
str(edx)
dim(edx) #number of rows and columns 
edx %>% slice(1:10) #example of the first 10 rows
names(edx) #names of the heading 

n_distinct(edx$movieId) #distinct number of movies
n_distinct(edx$userId) #distinct number of users 

# The summary of ratings by each category 
edx %>% 
  select(rating) %>% 
  group_by(rating) %>% 
  summarise(frequencies = n())

# Sampling of 100 movies and 100 userId to check the extent of NA value - code is borrowed from Prof. 
users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  dplyr::select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>% 
  (\(mat) mat[, sample(ncol(mat), 100)])()%>%
  as.matrix() %>% 
  t() %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# graphic representation of count of movies (graph_1) and those who rated (graph_2)
library(gridExtra)
graph_1 <- edx %>% count(movieId) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() + 
  ggtitle("Movies")

graph_2 <- edx %>% count(userId) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() + 
  ggtitle("Users")

grid.arrange(graph_1, graph_2, ncol = 2)

# numeric representation of count of movies (p1) and those who rated (p2)
edx %>% count(movieId) %>% summarise(mean = mean(n), min = min(n), Q1 = quantile(n, 0.25), median = median(n), Q3 = quantile(n, 0.75),  max= max(n))

edx %>% count(userId) %>% summarise(mean = mean(n), min = min(n), Q1 = quantile(n, 0.25), median = median(n), Q3 = quantile(n, 0.75),  max= max(n))


# RMSE calculation function: 
RMSE <- function(rating_Y, ratings_P){
  sqrt(mean((rating_Y - ratings_P)^2)) #Y= true, P=predicted
}

## 1st Model
mu <- mean(edx$rating) #this is mu_hat
mu #3.512465

model_1_rmse <- RMSE(final_holdout_test$rating, mu)
model_1_rmse

rmse_results <- tibble(Method = "Model 1. Average", RMSE = model_1_rmse) 

knitr::kable(rmse_results)


## 2nd Model 

# mu is from earlier chunk
movie_mean <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#graph representation of the b_i (with 1.5 being the highest possible value given mean of 3.5, and -3.0 being the minimal given the lowest rating of 0.5)
edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) %>% 
  ggplot(aes(b_i)) +
  geom_histogram(bins = 30, color = "black")

# additional impact of b_i on RMSE 
rating_P <- mu + final_holdout_test %>% 
  left_join(movie_mean, by='movieId') %>%
  pull(b_i)

model_2_rmse <- RMSE(rating_P, final_holdout_test$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model 2. Movie Effect Model",
                                 RMSE = model_2_rmse ))
knitr::kable(rmse_results) 

## 3rd Model 

#individual rater average rating by counts
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# additional impact of b_u on RMSE 
user_mean <- edx %>% 
  left_join(movie_mean, by='movieId') %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

rating_P <- final_holdout_test %>% 
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  mutate(predicted = mu + b_i + b_u) %>%
  pull(predicted)

model_3_rmse <- RMSE(rating_P, final_holdout_test$rating)

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model 3. Movie + User Effects Model",  
                                 RMSE = model_3_rmse ))

knitr::kable(rmse_results) 

## 4th Model 

#create date column separated by "week"
edx <- edx %>% 
  mutate(date = as_datetime(timestamp)) %>% 
  mutate(date = floor_date(date, unit = "week")) #floor_date better than round_date for week specific effects 

#similar operation is needed for the test_set 
final_holdout_test <- final_holdout_test %>% 
  mutate(date = as_datetime(timestamp)) %>% 
  mutate(date = floor_date(date, unit = "week"))

#building the prediction model 

date_mean <- edx %>% 
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  group_by(date) %>%
  summarise(b_t = mean(rating - mu - b_i - b_u))

rating_P <- final_holdout_test %>% 
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  left_join(date_mean, by='date') %>%
  mutate(predicted = mu + b_i + b_u + b_t) %>%
  pull(predicted)

model_4_rmse <- RMSE(rating_P, final_holdout_test$rating)

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model 4. Movie + User + Date Effects Model",  
                                 RMSE = model_4_rmse ))

knitr::kable(rmse_results) 

## 5th Model 

#there is a clear effect of genres - please excuse the obscuration of the x-axis labels - this will be more clearly demonstrated in the next section. 
edx %>%
  group_by(genres) %>%
  summarize(mean_rating = mean(rating), n = n(), se = sd(rating)/sqrt(n), title=title[1]) %>% 
  filter(n>=1000)  %>% 
  mutate(genres = reorder(genres, mean_rating)) %>% #arrange(mean_rating) this does not work
  ggplot(aes(x = genres, y = mean_rating, ymin = mean_rating - 2*se, ymax = mean_rating + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#code for the model 

genre_mean <- edx %>% 
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  left_join(date_mean, by='date') %>%
  group_by(genres) %>%
  summarise(b_g = mean(rating - mu - b_i - b_u - b_t))

rating_P <- final_holdout_test %>% 
  left_join(movie_mean, by='movieId') %>%
  left_join(user_mean, by='userId') %>%
  left_join(date_mean, by='date') %>%
  left_join(genre_mean, by='genres') %>%
  mutate(predicted = mu + b_i + b_u + b_t + b_g) %>%
  pull(predicted)

model_5_rmse <- RMSE(rating_P, final_holdout_test$rating)

rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Model 5. Movie + User + Date + Genre Effects Model",  
                                 RMSE = model_5_rmse ))

knitr::kable(rmse_results) 


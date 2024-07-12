library(haven)
library(tidyverse)
library(readxl)
library(GGally)
library(ggplot2)
install.packages("maps")
library(janitor)
library(car)
library(maps)
library(shiny)

#2022

df <- read_excel("WHR.xls")
head(df)

##data cleaning
df <- df %>% 
  drop_na() %>% 
  clean_names()


##descriptive statistics
summary(df)
ggpairs(select(df, -country_name, -year))

#train and test data
set.seed(24)
train = sample(1:nrow(df), 0.7*nrow(df))
test = df[-train,]
train = df[train,]

#model
lm1 <- lm(life_ladder~. -year -country_name, data=train)

summary(lm1)

par(mfrow=c(2,2))
plot(lm1)

vif(lm1)

probs <- predict(lm1, newdata = test)
test$probs <- probs

test <- test %>% 
  select(life_ladder, probs) %>% 
  mutate(diff = life_ladder-probs)


#model
df_test <- df %>% 
  mutate(social_support=social_support**2, healthy_life_expectancy_at_birth=healthy_life_expectancy_at_birth**2,
         perceptions_of_corruption=perceptions_of_corruption**2)


lm2 <- lm(life_ladder~ log_gdp_per_capita +  I(social_support^2) + I(healthy_life_expectancy_at_birth^2) + 
           freedom_to_make_life_choices + generosity + I(perceptions_of_corruption^2) + positive_affect, data=train)
summary(lm2)

par(mfrow=c(2,2))
plot(lm2)

ggpairs(select(df_test, -country_name, -year))

###
### EDA

as.factor(df$country_name) %>% levels()

df$country_name[df$country_name=="United States"]  <- "USA"
df$country_name[df$country_name=="United Kingdom"]  <- "UK"

world_map <- map_data("world")
unique(world_map$region)

full.map.data <- left_join(world_map, df, by=c("region" = "country_name"))


dev.off()
ggplot() + 
  geom_polygon(data = full.map.data, aes(x = long, y = lat, group=group, fill = life_ladder))


full.map.data2 <- filter(full.map.data, year==2012 | is.na(year))

full.map.data3 <- left_join(world_map, full.map.data2, by=c("region" = "region", "lat"='lat', 'long'='long', 'group'='group', 'order'='order'))





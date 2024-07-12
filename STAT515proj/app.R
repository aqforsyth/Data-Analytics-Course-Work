#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)
library(haven)
library(tidyverse)
library(readxl)
library(GGally)
library(ggplot2)
library(janitor)
library(car)
library(maps)
library(bslib)
library(plotly)
library(magrittr)
library(forcats)

# Get the data

df <- read_excel("WHR.xls")
df <- df %>% 
  drop_na() %>% 
  clean_names() %>% 
  filter(year!=2005)

df$country_name[df$country_name=="United States"]  <- "USA"
df$country_name[df$country_name=="United Kingdom"]  <- "UK"

world_map <- map_data("world")
unique(world_map$region)

full.map.data <- left_join(world_map, df, by=c("region" = "country_name"))


# Define UI

ui <- navbarPage(title = "World Happiness Report Data Exploration",
                 
                 tabPanel("About",
                          h4("this text explains the data")
                          ),
                 
                 tabPanel("Top and Bottom Ranked Countries By Year",
                          h3(textOutput("title1")),
                          h6("The Following graphics disply the top and bottom ranked countries for the year and variable selcted. The yearly average for all countries is displayed by the dashed red line."),
                          
                          page_sidebar(
                            
                            sidebar = sidebar(
                              # Select variable for y-axis
                              selectInput(
                                inputId = "y",
                                label = "Variable:",
                                choices = c("life_ladder","generosity", "positive_affect", "negative_affect", "log_gdp_per_capita", "social_support",
                                            "healthy_life_expectancy_at_birth", "perceptions_of_corruption", "freedom_to_make_life_choices"),
                                selected = "life_ladder"
                              ),
                              # Select variable for n
                              sliderInput(
                                inputId = "top",
                                label = "Rank N:",
                                min = 3,
                                max = 20,
                                value = 10
                              ),
                              # Select year
                              sliderInput(
                                inputId = "year",
                                label = "Year:",
                                min = 2006,
                                max = 2023,
                                value = 2023
                              )
                            ),
                            
                            #show value boxes
                            layout_column_wrap(
                              value_box(
                                title = "Minimum",
                                value = textOutput("min1"),
                                showcase = NULL,
                                theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                              ),
                              value_box(
                                title = "Average",
                                value = textOutput("avg1"),
                                showcase = NULL,
                                theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                              ),
                              value_box(
                                title = "Maximum",
                                value = textOutput("max1"),
                                showcase = NULL,
                                theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                              )
                            ),
                            
                            # Output: Show barplot
                            card(plotOutput(outputId = "barplot")),
                            card(plotOutput(outputId = "barplot2"))
                            )
                          ),
                 
                 tabPanel( "Variable distributions over the Years",
                   h3(textOutput("title2")),        
                   h6("The Following graphics disply the top and bottom ranked countries for the year and variable selcted. The yearly average for all countries is displayed by the dashed red line."),
                   
                   page_sidebar(
                     
                     sidebar = sidebar(
                       # Select variable for y-axis
                       selectInput(
                         inputId = "y2",
                         label = "Variable:",
                         choices = c("life_ladder","generosity", "positive_affect", "negative_affect", "log_gdp_per_capita", "social_support",
                                     "healthy_life_expectancy_at_birth", "perceptions_of_corruption", "freedom_to_make_life_choices"),
                         selected = "life_ladder"
                       )
                     ),
                     
                     #show value boxes
                     layout_column_wrap(
                       value_box(
                         title = "Change in Average Since 2006",
                         value = textOutput("change2"),
                         showcase = NULL,
                         theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                       ),
                       value_box(
                         title = "Mean of all Data",
                         value = textOutput("avg2"),
                         showcase = NULL,
                         theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                       ),
                       value_box(
                         title = "Median of all Data",
                         value = textOutput("med2"),
                         showcase = NULL,
                         theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                       )
                     ),
                     
                     # Output: Show boxplot
                     card(plotOutput(outputId = "boxplot"))
                   )
                 ),
                 
                 tabPanel( "World Map",
                   h3(textOutput("title3")),        
                   h6("This Page displays a world map of countries in the World Happiness data report"),
                   
                   page_sidebar(
                     
                     sidebar = sidebar(
                       # Select variable for
                       selectInput(
                         inputId = "y3",
                         label = "Variable:",
                         choices = c("life_ladder","generosity", "positive_affect", "negative_affect", "log_gdp_per_capita", "social_support",
                                     "healthy_life_expectancy_at_birth", "perceptions_of_corruption", "freedom_to_make_life_choices"),
                         selected = "life_ladder"
                       ),
                       
                       selectInput(
                         inputId = "country",
                         label = "Country",
                         choices = unique(df$country_name),
                         selected = "USA"
                       ),
                       
                       # Select year
                       sliderInput(
                         inputId = "year2",
                         label = "Year:",
                         min = 2006,
                         max = 2023,
                         value = 2023
                       )
                     ),
                     
                     layout_column_wrap(
                       value_box(
                         title = "Value for Selected Country",
                         value = textOutput("val3"),
                         showcase = NULL,
                         theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                       ),
                       value_box(
                         title = "Median",
                         value = textOutput("med3"),
                         showcase = NULL,
                         theme = value_box_theme(bg = "#104e8b", fg = "#EBF4F6")
                       )
                     ),
                     
                     # Output: Show map
                     card(height= 600, full_screen = TRUE, plotOutput(outputId = "map")),
                     card(full_screen = TRUE, plotOutput(outputId = "line"))
                   )
                   
                 ),
                 
                 tabPanel("Data", tableOutput("data"))
                 
                )


# Define server

server <- function(input, output, session) {
  output$barplot <- renderPlot({
    
    plot_title <- paste0("Top ", input$top, " Countries With the Highest ",  input$y, " Score in ", input$year)
    
    yr_avg <- df %>% 
      filter(year==input$year) %>%
      pull(!!rlang::sym(input$y)) %>% 
      mean()
    
    df2 <- df %>% 
      filter(year==input$year) %>% 
      dplyr::arrange(dplyr::desc(!!rlang::sym(input$y))) %>% 
      head(input$top) %>% 
      mutate(country_name = fct_reorder(country_name, desc(!!rlang::sym(input$y))))
    
    
    ggplot(data = df2, aes_string(x = as.factor(df2$country_name), y = input$y)) +
      geom_bar(stat="identity", fill="#088395")+
      labs(title = plot_title,
           x= "Country")+
      theme_minimal()+
      geom_hline(yintercept=yr_avg, linetype="dashed", color = "red")
  })
  
  output$barplot2 <- renderPlot({
    
    plot_title <- paste0("Bottom ", input$top, " Countries With the Lowest ",  input$y, " Score in ", input$year)
    
    yr_avg <- df %>% 
      filter(year==input$year) %>%
      pull(!!rlang::sym(input$y)) %>% 
      mean()
    
    df2 <- df %>% 
      filter(year==input$year) %>% 
      dplyr::arrange(!!rlang::sym(input$y)) %>% 
      head(input$top) %>% 
      mutate(country_name = fct_reorder(country_name, desc(!!rlang::sym(input$y))))
    
    
    ggplot(data = df2, aes_string(x = as.factor(df2$country_name), y = input$y)) +
      geom_bar(stat="identity", fill="#37B7C3")+
      labs(title = plot_title,
           x= "Country")+
      theme_minimal()+
      geom_hline(yintercept=yr_avg, linetype="dashed", color = "red")
  })
  
  output$avg1 <- renderText({
    yr_avg <- df %>% 
      filter(year==input$year) %>%
      pull(!!rlang::sym(input$y)) %>% 
      mean()
  })
  
  output$max1 <- renderText({
    yr_avg <- df %>% 
      filter(year==input$year) %>%
      pull(!!rlang::sym(input$y)) %>% 
      max()
  })
  
  output$min1 <- renderText({
    yr_avg <- df %>% 
      filter(year==input$year) %>%
      pull(!!rlang::sym(input$y)) %>% 
      min()
  })
  
  output$title1 <- renderText({
    string <- sub("_", " ", input$y)
    paste(str_to_sentence(string))
  })
  
  output$data <- renderTable({
    df
  })
  
  output$boxplot <- renderPlot({
    
    plot_title <- paste0("Boxplot of ", input$y2, " from 2006-2023")
    
    ggplot(data = df, aes_string(x = as.factor(df$year), y = input$y2)) +
      geom_boxplot(fill="#EBF4F6")+
      labs(title = plot_title,
           x= "Year")+
      theme_minimal()
    
  })
  
  output$title2 <- renderText({
    string <- sub("_", " ", input$y2)
    paste(str_to_sentence(string))
  })
  
  output$avg2 <- renderText({
    avg <- df %>% 
      pull(!!rlang::sym(input$y2)) %>% 
      mean()
  })
  
  output$med2 <- renderText({
    med <- df %>% 
      pull(!!rlang::sym(input$y2)) %>%
      median()
  })
  
  output$change2 <- renderText({
    min2006 <- df %>% 
      filter(year==2006) %>%
      pull(!!rlang::sym(input$y2)) %>%
      mean()
    max2023 <- df %>% 
      filter(year==2023) %>%
      pull(!!rlang::sym(input$y2)) %>%
      mean()
    print(max2023-min2006)
    
  })
  
  output$title3 <- renderText({
    string <- sub("_", " ", input$y3)
    paste(str_to_sentence(string))
  })

  output$map <- renderPlot({
    plot_title <- paste0("World Map of ", input$y3, " in ", input$year2)
    
    # Set colors
    full.map.data2 <- mutate(full.map.data, color1 = ifelse(region %in% c(input$country), "red", "darkgrey"))
    full.map.data2 <- filter(full.map.data2, year==input$year2 | is.na(year))
    
    map_dat <- left_join(world_map, full.map.data2, by=c("region" = "region", "lat"='lat', 'long'='long', 'group'='group', 'order'='order'))
    
    ggplot(data= map_dat,aes(colour = color1)) + 
      geom_polygon(aes_string(x = map_dat$long, y = map_dat$lat,
                                                    group=map_dat$group, fill = input$y3))+
      labs(title= plot_title,
           x = "Longitude",
           y = "Lattitude")+
      theme_minimal()+
      scale_colour_identity()
    
  })
  
  output$line <- renderPlot({
    
    yr_avg <- df %>% 
      group_by(year) %>% 
      summarise(mean= mean(!!rlang::sym(input$y3)))
    
    plot_title <- paste0("Line Chart of ", input$y3, " for ", input$country, " from 2006-2023")
    
    df3 <- df %>% 
      filter(country_name==input$country)
    
    ggplot() +
      geom_line(data = df3, aes_string(x = df3$year, y = input$y3),color="#104e8b", size=2)+
      geom_line(data = yr_avg, aes_string(x = yr_avg$year, y = yr_avg$mean),color="red", size=1, linetype="dashed")+
      labs(title = plot_title,
           x= "Year",
           y= input$y3,
           subtitle = "Country data shown in blue, world average shown in red dashed line")+
      theme_minimal()
    
  })
  
  output$med3 <- renderText({
    med3 <- df %>% 
      filter(year==input$year2) %>% 
      pull(!!rlang::sym(input$y3)) %>%
      median()
  })
  
  output$val3 <- renderText({
    val <- df %>% 
      filter(country_name == input$country) %>% 
      filter(year==input$year2) %>%
      pull(!!rlang::sym(input$y3)) %>%
      median()
  })
  

}

# Create a Shiny app

shinyApp(ui = ui, server = server)






Greta and Causact Exploration
================
Peter Rabinovitch
2021-12-27 14:35:41

# Introduction

The purpose of this document is to illustrate how to use the *causact*
library to build some simple models. Causact allows you to create
Bayesian models, and uses the *greta* library to do the computations.
Well, not really - greta just sets them up, and uses TensorFlow
underneath.

There are already some good resources for both
[greta](https://greta-stats.org/index.html) &
[causact](https://www.causact.com/), but I wanted to try some simple
examples, and figured others might find this useful too.

To get everything set up, you can do no better than to follow [chapter
15 of “A Business Analyst’s Introduction to Business
Analytics”](https://www.causact.com/install-tensorflow-greta-and-causact.html#install-tensorflow-greta-and-causact)

``` r
library(tidyverse)
library(tictoc)
library(lubridate)
library(skimr)
library(patchwork)
library(ggridges)
library(knitr)
library(DT)
library(janitor)
library(broom)
library(tictoc)
library(lme4)
library(bayesplot)
library(greta)
library(causact)

set.seed(2021)

g_fignum <- 0
figcap <- function() {
  g_fignum <<- g_fignum + 1
  return(str_c("Figure ", g_fignum))
}

g_tabnum <- 0
tabcap <- function() {
  g_tabnum <<- g_tabnum + 1
  return(str_c("Table ", g_tabnum))
}
```

Note that this document will not discuss all the things you need to do
to perform a proper Bayesian analysis, that is discussed elsewhere (see
the references). The purpose here is simply to show how to use causact &
greta.

# Data

In order to build some simple examples we will work with one data set.

``` r
subjectdf <- tribble(
  ~condition, ~covariate, ~subject,
  "treatment", "b", "anbkv",
  "control", "b", "avpov",
  "control", "b", "ayyxo",
  "treatment", "a", "bkydk",
  "treatment", "b", "brcyb",
  "treatment", "b", "dcrhb",
  "treatment", "a", "efsoy",
  "treatment", "b", "emjwt",
  "treatment", "b", "ewlyd",
  "control", "a", "fryng",
  "treatment", "b", "fygyl",
  "control", "a", "fzvmk",
  "treatment", "a", "gjleq",
  "control", "b", "gnwyo",
  "treatment", "a", "ifxpf",
  "control", "a", "ihawn",
  "treatment", "a", "jzmgc",
  "control", "a", "lmhsz",
  "control", "b", "lusjq",
  "treatment", "a", "mbgbb",
  "treatment", "a", "nglsq",
  "treatment", "a", "ofrrl",
  "control", "a", "okyrj",
  "treatment", "b", "orbnj",
  "control", "b", "sqgvz",
  "treatment", "b", "uxabv",
  "control", "a", "xqsts",
  "control", "b", "xywyv"
)

resultsdf <- tribble(
  ~time, ~anbkv, ~avpov, ~ayyxo, ~bkydk, ~brcyb, ~dcrhb, ~efsoy, ~emjwt, ~ewlyd, ~fryng, ~fygyl, ~fzvmk, ~gjleq, ~gnwyo, ~ifxpf, ~ihawn, ~jzmgc, ~lmhsz, ~lusjq, ~mbgbb, ~nglsq, ~ofrrl, ~okyrj, ~orbnj, ~sqgvz, ~uxabv, ~xqsts, ~xywyv,
  1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
  2L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
  3L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
  4L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
  5L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
  6L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
  7L, 0L, 0L, 0L, 0L, 0L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
  8L, 0L, 0L, 0L, 0L, 0L, 2L, 0L, 0L, 0L, 3L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 1L, 0L, 0L, 1L, 0L, 0L, 0L, 0L, 0L,
  9L, 2L, 1L, 0L, 0L, 2L, 2L, 2L, 2L, 0L, 3L, 2L, 3L, 0L, 2L, 0L, 0L, 0L, 0L, 2L, 2L, 0L, 1L, 2L, 0L, 0L, 1L, 0L, 0L,
  10L, 3L, 3L, 0L, 2L, 3L, 2L, 2L, 2L, 2L, 3L, 3L, 3L, 2L, 3L, 2L, 1L, 0L, 0L, 3L, 3L, 2L, 2L, 3L, 0L, 2L, 3L, 1L, 1L,
  11L, 3L, 3L, 1L, 2L, 4L, 3L, 3L, 2L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 1L, 0L, 3L, 3L, 3L, 3L, 3L, 2L, 2L, 3L, 2L, 2L,
  12L, 3L, 3L, 3L, 3L, 5L, 3L, 3L, 3L, 3L, 3L, 3L, 2L, 3L, 3L, 3L, 3L, 3L, 0L, 3L, 3L, 3L, 3L, 3L, 3L, 2L, 3L, 2L, 3L,
  13L, 4L, 3L, 3L, 3L, 5L, 2L, 4L, 3L, 4L, 3L, 4L, 2L, 3L, 3L, 3L, 3L, 3L, 1L, 3L, 3L, 2L, 2L, 3L, 3L, 2L, 3L, 3L, 2L,
  14L, 4L, 3L, 3L, 3L, 5L, 2L, 4L, 3L, 3L, 2L, 3L, 2L, 3L, 3L, 3L, 3L, 3L, 2L, 3L, 3L, 2L, 2L, 3L, 3L, 2L, 4L, 3L, 2L,
  15L, 4L, 3L, 3L, 3L, 5L, 2L, 4L, 3L, 3L, 3L, 3L, 2L, 3L, 2L, 3L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 3L, 2L, 4L, 3L, 2L,
  16L, 5L, 2L, 3L, 3L, 5L, 2L, 4L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 3L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 3L, 2L,
  17L, 5L, 2L, 2L, 3L, 5L, 2L, 4L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L,
  18L, 5L, 2L, 2L, 3L, 5L, 2L, 4L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L,
  19L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L,
  20L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 1L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L,
  21L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L,
  22L, 5L, 1L, 2L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 1L, 2L, 2L, 2L, 2L, 3L, 1L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L,
  23L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 3L, 2L, 2L,
  24L, 5L, 2L, 3L, 2L, 5L, 2L, 5L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 3L, 2L, 2L,
  25L, 5L, 2L, 3L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L,
  26L, 5L, 2L, 3L, 2L, 5L, 2L, 5L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L,
  27L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L,
  28L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 3L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L,
  29L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L,
  30L, 5L, 2L, 2L, 2L, 5L, 2L, 5L, 2L, 3L, 2L, 3L, 2L, 2L, 2L, 2L, 2L, 3L, 0L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L
)

resultsdfl <- resultsdf %>%
  pivot_longer(
    cols = -time,
    names_to = "subject",
    values_to = "outcome"
  )

df <- resultsdfl %>%
  left_join(subjectdf, by = c("subject" = "subject")) %>%
  arrange(subject, time)

rm(resultsdfl, resultsdf, subjectdf)
```

# EDA

Ok, so what does this data set look like?

``` r
df %>% glimpse()
```

    ## Rows: 840
    ## Columns: 5
    ## $ time      <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1…
    ## $ subject   <chr> "anbkv", "anbkv", "anbkv", "anbkv", "anbkv", "anbkv", "anbkv…
    ## $ outcome   <int> 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, …
    ## $ condition <chr> "treatment", "treatment", "treatment", "treatment", "treatme…
    ## $ covariate <chr> "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", …

We have five fields:

-   subject: a unique id for each subject.

-   outcome: the outcome at a particular time. It is on a scale of 0-5.

-   condition: whether the subject was in the control or treatment
    (experimental) group

-   covariate: a covariate relating to the subject. In this data set
    there are only to possibilities, *a* and *b*.

-   time: the time at which the outcome was observed.

A few fictitious examples will help clarify:

-   perhaps the subjects are people, the outcome is the process of a
    disease, the condition is whether they have been given a placebo or
    an experimental medicine, the covariate could be to which group the
    subject belong (eg gender, adult or senior, etc) and the time could
    be the number of days after the experiment has started.

-   in another example the subjects are stock portfolios, the outcome is
    the growth of the portfolio value, the condition is whether the
    portfolio is using a standard algorithm to select stocks or a new
    experimental one, the covariate could be which exchange the stocks
    are purchased on and the time could be the number of weeks after the
    experiment has started.

Obviously many more examples could be proposed, but the point is that
this is a very general setup, but still presents interesting statistical
challenges.

For one example, a linear regression is not appropriate here because all
errors for a subject are correlated.

Let us skim the dataset:

``` r
df %>% skim()
```

|                                                  |            |
|:-------------------------------------------------|:-----------|
| Name                                             | Piped data |
| Number of rows                                   | 840        |
| Number of columns                                | 5          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |            |
| Column type frequency:                           |            |
| character                                        | 3          |
| numeric                                          | 2          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |            |
| Group variables                                  | None       |

Data summary

**Variable type: character**

| skim_variable | n_missing | complete_rate | min | max | empty | n_unique | whitespace |
|:--------------|----------:|--------------:|----:|----:|------:|---------:|-----------:|
| subject       |         0 |             1 |   5 |   5 |     0 |       28 |          0 |
| condition     |         0 |             1 |   7 |   9 |     0 |        2 |          0 |
| covariate     |         0 |             1 |   1 |   1 |     0 |        2 |          0 |

**Variable type: numeric**

| skim_variable | n_missing | complete_rate |  mean |   sd |  p0 | p25 |  p50 | p75 | p100 | hist  |
|:--------------|----------:|--------------:|------:|-----:|----:|----:|-----:|----:|-----:|:------|
| time          |         0 |             1 | 15.50 | 8.66 |   1 |   8 | 15.5 |  23 |   30 | ▇▇▇▇▇ |
| outcome       |         0 |             1 |  1.81 | 1.39 |   0 |   0 |  2.0 |   3 |    5 | ▆▇▅▁▁ |

and do a first plot:

``` r
df %>%
  ggplot(aes(x = time, y = outcome, colour = subject)) +
  geom_line() +
  theme_minimal() +
  facet_wrap(covariate ~ condition) +
  theme(legend.position = "none")
```

![Figure
1](greta_causact_exploration_files/figure-gfm/unnamed-chunk-5-1.png)
From these plots we see that the outcome is not a linear function of
time, and hence we may be better off treating time as a qualitative,
rather than quantitative variable.

``` r
df <- df %>% mutate(timec = as.character(time))
```

Lets see if the maximum outcome differs by condition or covariate:

``` r
df %>%
  group_by(condition, covariate) %>%
  summarize(max_outcome = max(outcome)) %>%
  pivot_wider(names_from = covariate, values_from = max_outcome)
```

    ## # A tibble: 2 × 3
    ## # Groups:   condition [2]
    ##   condition     a     b
    ##   <chr>     <int> <int>
    ## 1 control       3     3
    ## 2 treatment     5     5

We see that under the treatment, both covariate groups have a higher
maximum outcome than the control group.

# Simple Models

In this section we will build some basic models using causact, and in
some cases we will explore further with greta, or by doing various
plots. Causact & greta play nicely with many of the typical Bayes
packages (eg bayesplot), so if you have a favourite, try it out.

One thing to note is that causact, greta and tensorflow all respect the
set.seed, so your results will be reproducible.

For each model we title it by a similar lm.

## Model 1: lm(outcome\~1)

Here we will look at calculating just one mean outcome for the whole
dataset, regardless of time, covariate or condition.

First we build the model. We assume that the outcomes are normally
distributed with standard deviation 3, and a mean mu that is distributed
uniformly between 0 and 5. Obviously these are crappy priors, but the
purpose here is illustration of how to use causact, and some of greta.

You can do a simple model like so:

``` r
dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = uniform(0, 5), child = "o") %>%
  dag_render()
```

![Figure
2](greta_causact_exploration_files/figure-gfm/unnamed-chunk-8-1.png)

but it is frequently helpful to use the plate notation as in the
following.

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = uniform(0, 5), child = "o") %>%
  dag_plate("Observation", "i", nodeLabels = c("o"))
```

Now that we’ve built the model, we can plot it (not the results, the
model itself)!

``` r
graph %>% dag_render()
```

![Figure
3](greta_causact_exploration_files/figure-gfm/unnamed-chunk-10-1.png)

Next, we can estimate the parameters of this model, i.e. mu.

``` r
drawsDF <- graph %>% dag_greta()
```

and we can plot the estimates of the parameters

``` r
drawsDF %>% dagp_plot()
```

![Figure
4](greta_causact_exploration_files/figure-gfm/unnamed-chunk-12-1.png)

We can also plot some diagnostic info from the estimation procedure

``` r
mcmc_trace(drawsDF)
```

![Figure
5](greta_causact_exploration_files/figure-gfm/unnamed-chunk-13-1.png)

Since we have the samples in drawsDF, we can do anything we want with
them.

``` r
drawsDF %>% summary()
```

    ##        mu       
    ##  Min.   :1.480  
    ##  1st Qu.:1.743  
    ##  Median :1.812  
    ##  Mean   :1.813  
    ##  3rd Qu.:1.882  
    ##  Max.   :2.172

Another cool thing is that you can get the greta code from the causact
model:

``` r
graph %>% dag_greta(mcmc = FALSE)
```

    ## ## The below greta code will return a posterior distribution 
    ## ## for the given DAG. Either copy and paste this code to use greta
    ## ## directly, evaluate the output object using 'eval', or 
    ## ## or (preferably) use dag_greta(mcmc=TRUE) to return a data frame of
    ## ## the posterior distribution: 
    ## o <- as_data(df$outcome)   #DATA
    ## mu     <- uniform(min = 0, max = 5)   #PRIOR
    ## distribution(o) <- normal(mean = mu, sd = 3)   #LIKELIHOOD
    ## gretaModel  <- model(mu)   #MODEL
    ## meaningfulLabels(graph)
    ## draws       <- mcmc(gretaModel)              #POSTERIOR
    ## drawsDF     <- replaceLabels(draws) %>% as.matrix() %>%
    ##                 dplyr::as_tibble()           #POSTERIOR
    ## tidyDrawsDF <- drawsDF %>% addPriorGroups()  #POSTERIOR

This is helpful for debugging, if you need to. You can also run this
code:

``` r
o <- as_data(df$outcome) # DATA
mu <- uniform(min = 0, max = 5) # PRIOR
distribution(o) <- normal(mean = mu, sd = 3) # LIKELIHOOD
gretaModel <- model(mu) # MODEL
meaningfulLabels(graph)
draws <- mcmc(gretaModel) # POSTERIOR
drawsDF <- replaceLabels(draws) %>%
  as.matrix() %>%
  dplyr::as_tibble() # POSTERIOR
tidyDrawsDF <- drawsDF %>% addPriorGroups() # POSTERIOR
```

and then plot the greta model

``` r
gretaModel %>% plot()
```

![Figure
6](greta_causact_exploration_files/figure-gfm/unnamed-chunk-17-1.png)

which is a different representation of the causact model plotted
earlier.

All these diagrams ease communication between yourself & stakeholders,
as well as yourself in three week (or months, or…)

You can also do things like

``` r
draws %>% summary()
```

    ## 
    ## Iterations = 1:1000
    ## Thinning interval = 1 
    ## Number of chains = 4 
    ## Sample size per chain = 1000 
    ## 
    ## 1. Empirical mean and standard deviation for each variable,
    ##    plus standard error of the mean:
    ## 
    ##           Mean             SD       Naive SE Time-series SE 
    ##       1.805006       0.106956       0.001691       0.004545 
    ## 
    ## 2. Quantiles for each variable:
    ## 
    ##  2.5%   25%   50%   75% 97.5% 
    ## 1.591 1.734 1.807 1.878 2.008

``` r
mcmc_trace(draws)
```

![Figure
7](greta_causact_exploration_files/figure-gfm/unnamed-chunk-19-1.png)

``` r
mcmc_intervals(draws)
```

![Figure
8](greta_causact_exploration_files/figure-gfm/unnamed-chunk-20-1.png)

Moving back and forth between causact & greta ensures you build the
model you think you built, and that you can take advantage of all the
available tools. Very cool.

## Model 2: lm(outcome\~condition)

Here we will look at calculating one mean outcome for each condition,
regardless of time or covariate.

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = uniform(0, 5), child = "o") %>%
  dag_plate("Condition", "i", nodeLabels = c("mu"), data = df$condition, addDataNode = TRUE)
graph %>% dag_render()
```

![Figure
9](greta_causact_exploration_files/figure-gfm/unnamed-chunk-21-1.png)

``` r
drawsDF <- graph %>% dag_greta()
```

``` r
drawsDF %>% dagp_plot()
```

![Figure
10](greta_causact_exploration_files/figure-gfm/unnamed-chunk-22-1.png)

## Model 3: lm(outcome\~covariate\*condition)

Here we will look at calculating just one mean outcome per
condition-covariate combination, regardless of time.

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = uniform(0, 5), child = "o") %>%
  dag_plate("Condition Effect", "i", nodeLabels = c("mu"), data = df$condition, addDataNode = TRUE) %>%
  dag_plate("Covariate Effect", "j", nodeLabels = c("mu"), data = df$covariate, addDataNode = TRUE)
graph %>% dag_render()
```

![Figure
11](greta_causact_exploration_files/figure-gfm/unnamed-chunk-23-1.png)

``` r
drawsDF <- graph %>% dag_greta()
drawsDF %>% dagp_plot()
```

![Figure
12](greta_causact_exploration_files/figure-gfm/unnamed-chunk-24-1.png)

## Model 4: lm(outcome\~covariate+condition)

Here we will look at calculating just one mean outcome per condition,
and one mean outcome per covariate.

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = condeffect + coveffect, child = "o") %>%
  dag_node("condeffect", "condeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("coveffect", "coveffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_plate("Condition Effect", "i", nodeLabels = c("condeffect"), data = df$condition, addDataNode = TRUE) %>%
  dag_plate("Covariate Effect", "j", nodeLabels = c("coveffect"), data = df$covariate, addDataNode = TRUE)
graph %>% dag_render()
```

![Figure
13](greta_causact_exploration_files/figure-gfm/unnamed-chunk-25-1.png)

``` r
drawsDF <- graph %>% dag_greta()
drawsDF %>% dagp_plot()
```

![Figure
14](greta_causact_exploration_files/figure-gfm/unnamed-chunk-26-1.png)

## Model 5: lm(outcome\~covariate+condition+timec)

Here we will look at calculating just one mean outcome per condition,
and one mean outcome per covariate.

Note we use timec here as we previously said, the outcome is definitely
not linear in time.

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = condeffect + coveffect + timeeffect, child = "o") %>%
  dag_node("condeffect", "condeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("coveffect", "coveffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("timeeffect", "timeeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_plate("Condition Effect", "i", nodeLabels = c("condeffect"), data = df$condition, addDataNode = TRUE) %>%
  dag_plate("Covariate Effect", "j", nodeLabels = c("coveffect"), data = df$covariate, addDataNode = TRUE) %>%
  dag_plate("Time Effect", "t", nodeLabels = c("timeeffect"), data = df$timec, addDataNode = TRUE)
graph %>% dag_render()
```

![Figure
15](greta_causact_exploration_files/figure-gfm/unnamed-chunk-27-1.png)

``` r
drawsDF <- graph %>% dag_greta()
drawsDF %>% dagp_plot()
```

![Figure
16](greta_causact_exploration_files/figure-gfm/unnamed-chunk-28-1.png)

## Model 6: lm(outcome\~covariate+condition+timec+subject)

one condition effect plus one covariate effect plus one time effect plus
one subject effect

Here we will look at calculating one mean outcome per condition, one
mean outcome per covariate, one mean outcome per time period, and one
per subject

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = condeffect + coveffect + timeeffect + subeffect, child = "o") %>%
  dag_node("condeffect", "condeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("coveffect", "coveffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("timeeffect", "timeeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("subeffect", "subeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_plate("Condition Effect", "i", nodeLabels = c("condeffect"), data = df$condition, addDataNode = TRUE) %>%
  dag_plate("Covariate Effect", "j", nodeLabels = c("coveffect"), data = df$covariate, addDataNode = TRUE) %>%
  dag_plate("Time Effect", "t", nodeLabels = c("timeeffect"), data = df$timec, addDataNode = TRUE) %>%
  dag_plate("Subject Effect", "s", nodeLabels = c("subeffect"), data = df$subject, addDataNode = TRUE)
graph %>% dag_render()
```

![Figure
17](greta_causact_exploration_files/figure-gfm/unnamed-chunk-29-1.png)

``` r
drawsDF <- graph %>% dag_greta()
drawsDF %>% dagp_plot()
```

![Figure
18](greta_causact_exploration_files/figure-gfm/unnamed-chunk-30-1.png)

## Model 7: lm(outcome\~covariate+condition+timec+subject) with plates

This is just like model 6, but we use the plate notation for the
outcome.

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df$outcome) %>%
  dag_node("mu", "mu", rhs = condeffect + coveffect + timeeffect + subeffect, child = "o") %>%
  dag_node("condeffect", "condeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("coveffect", "coveffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("timeeffect", "timeeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("subeffect", "subeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_plate("Condition Effect", "i", nodeLabels = c("condeffect"), data = df$condition, addDataNode = TRUE) %>%
  dag_plate("Covariate Effect", "j", nodeLabels = c("coveffect"), data = df$covariate, addDataNode = TRUE) %>%
  dag_plate("Time Effect", "t", nodeLabels = c("timeeffect"), data = df$timec, addDataNode = TRUE) %>%
  dag_plate("Subject Effect", "s", nodeLabels = c("subeffect"), data = df$subject, addDataNode = TRUE) %>%
  dag_plate("Observation", "b", nodeLabels = c("o"))
graph %>% dag_render()
```

![Figure
19](greta_causact_exploration_files/figure-gfm/unnamed-chunk-31-1.png)

``` r
drawsDF <- graph %>% dag_greta()
drawsDF %>% dagp_plot()
```

![Figure
20](greta_causact_exploration_files/figure-gfm/unnamed-chunk-32-1.png)

Here we show how we can use the samples to plot other things of
interest.

First we look at the samples

``` r
drawsDF %>% glimpse()
```

    ## Rows: 4,000
    ## Columns: 62
    ## $ condeffect_control  <dbl> 0.0201227063, 0.0201227063, 0.0020600005, 0.002060…
    ## $ condeffect_treatmnt <dbl> 0.1135729021, 0.1135729021, 0.0546426988, 0.054642…
    ## $ coveffect_a         <dbl> 0.0003525506, 0.0003525506, 0.0019178713, 0.001917…
    ## $ coveffect_b         <dbl> 0.0447572935, 0.0447572935, 0.0449158460, 0.044915…
    ## $ timeeffect_1        <dbl> 0.25742781, 0.25742781, 0.21180499, 0.21180499, 0.…
    ## $ timeeffect_10       <dbl> 0.5412902, 0.5412902, 0.7466917, 0.7466917, 0.7466…
    ## $ timeeffect_11       <dbl> 2.1955580, 2.1955580, 1.8447240, 1.8447240, 1.8447…
    ## $ timeeffect_12       <dbl> 2.1061552, 2.1061552, 1.5578484, 1.5578484, 1.5578…
    ## $ timeeffect_13       <dbl> 2.607180, 2.607180, 2.101818, 2.101818, 2.101818, …
    ## $ timeeffect_14       <dbl> 2.375174, 2.375174, 2.382853, 2.382853, 2.382853, …
    ## $ timeeffect_15       <dbl> 2.343494, 2.343494, 1.614113, 1.614113, 1.614113, …
    ## $ timeeffect_16       <dbl> 1.5871510, 1.5871510, 1.0149024, 1.0149024, 1.0149…
    ## $ timeeffect_17       <dbl> 1.6065563, 1.6065563, 1.2088409, 1.2088409, 1.2088…
    ## $ timeeffect_18       <dbl> 1.6441981, 1.6441981, 1.7599420, 1.7599420, 1.7599…
    ## $ timeeffect_19       <dbl> 2.3797626, 2.3797626, 2.5541554, 2.5541554, 2.5541…
    ## $ timeeffect_2        <dbl> 0.055162446, 0.055162446, 0.096417989, 0.096417989…
    ## $ timeeffect_20       <dbl> 1.9185490, 1.9185490, 1.2586583, 1.2586583, 1.2586…
    ## $ timeeffect_21       <dbl> 1.4845784, 1.4845784, 1.8022967, 1.8022967, 1.8022…
    ## $ timeeffect_22       <dbl> 1.5658860, 1.5658860, 1.6781527, 1.6781527, 1.6781…
    ## $ timeeffect_23       <dbl> 1.779743, 1.779743, 1.661045, 1.661045, 1.661045, …
    ## $ timeeffect_24       <dbl> 1.2903848, 1.2903848, 1.2173554, 1.2173554, 1.2173…
    ## $ timeeffect_25       <dbl> 1.7392289, 1.7392289, 0.9622087, 0.9622087, 0.9622…
    ## $ timeeffect_26       <dbl> 2.6280718, 2.6280718, 0.6786904, 0.6786904, 0.6786…
    ## $ timeeffect_27       <dbl> 1.9183457, 1.9183457, 0.8955999, 0.8955999, 0.8955…
    ## $ timeeffect_28       <dbl> 1.1515044, 1.1515044, 1.2128844, 1.2128844, 1.2128…
    ## $ timeeffect_29       <dbl> 1.5518980, 1.5518980, 0.9852473, 0.9852473, 0.9852…
    ## $ timeeffect_3        <dbl> 0.24404571, 0.24404571, 0.05058443, 0.05058443, 0.…
    ## $ timeeffect_30       <dbl> 2.7226404, 2.7226404, 2.7886270, 2.7886270, 2.7886…
    ## $ timeeffect_4        <dbl> 0.19377647, 0.19377647, 0.17871182, 0.17871182, 0.…
    ## $ timeeffect_5        <dbl> 0.001998687, 0.001998687, 0.104054976, 0.104054976…
    ## $ timeeffect_6        <dbl> 0.40234132, 0.40234132, 0.29266718, 0.29266718, 0.…
    ## $ timeeffect_7        <dbl> 0.30389731, 0.30389731, 0.35936528, 0.35936528, 0.…
    ## $ timeeffect_8        <dbl> 0.36088341, 0.36088341, 0.27980501, 0.27980501, 0.…
    ## $ timeeffect_9        <dbl> 0.11612478, 0.11612478, 0.04325825, 0.04325825, 0.…
    ## $ subeffect_anbkv     <dbl> 1.585716, 1.585716, 2.696109, 2.696109, 2.696109, …
    ## $ subeffect_avpov     <dbl> 0.6907656, 0.6907656, 1.3474485, 1.3474485, 1.3474…
    ## $ subeffect_ayyxo     <dbl> 0.3874890, 0.3874890, 0.6893037, 0.6893037, 0.6893…
    ## $ subeffect_bkydk     <dbl> 1.1164892, 1.1164892, 0.6306231, 0.6306231, 0.6306…
    ## $ subeffect_brcyb     <dbl> 1.9317834, 1.9317834, 2.2085572, 2.2085572, 2.2085…
    ## $ subeffect_dcrhb     <dbl> 0.2878430, 0.2878430, 0.2717332, 0.2717332, 0.2717…
    ## $ subeffect_efsoy     <dbl> 1.4841184, 1.4841184, 2.0201185, 2.0201185, 2.0201…
    ## $ subeffect_emjwt     <dbl> 0.07891784, 0.07891784, 0.17568894, 0.17568894, 0.…
    ## $ subeffect_ewlyd     <dbl> 0.09435449, 0.09435449, 1.97980375, 1.97980375, 1.…
    ## $ subeffect_fryng     <dbl> 0.6082566, 0.6082566, 1.0709007, 1.0709007, 1.0709…
    ## $ subeffect_fygyl     <dbl> 0.93268538, 0.93268538, 1.16187955, 1.16187955, 1.…
    ## $ subeffect_fzvmk     <dbl> 0.39846638, 0.39846638, 0.71836686, 0.71836686, 0.…
    ## $ subeffect_gjleq     <dbl> 0.43980131, 0.43980131, 1.27060047, 1.27060047, 1.…
    ## $ subeffect_gnwyo     <dbl> 0.39527694, 0.39527694, 0.49524294, 0.49524294, 0.…
    ## $ subeffect_ifxpf     <dbl> 1.1397055, 1.1397055, 0.8237472, 0.8237472, 0.8237…
    ## $ subeffect_ihawn     <dbl> 0.38419044, 0.38419044, 0.40317782, 0.40317782, 0.…
    ## $ subeffect_jzmgc     <dbl> 0.15308811, 0.15308811, 1.22658716, 1.22658716, 1.…
    ## $ subeffect_lmhsz     <dbl> 0.46948662, 0.46948662, 0.03730992, 0.03730992, 0.…
    ## $ subeffect_lusjq     <dbl> 0.1320012, 0.1320012, 0.3709945, 0.3709945, 0.3709…
    ## $ subeffect_mbgbb     <dbl> 1.18773946, 1.18773946, 0.70771032, 0.70771032, 0.…
    ## $ subeffect_nglsq     <dbl> 0.49878585, 0.49878585, 0.14670421, 0.14670421, 0.…
    ## $ subeffect_ofrrl     <dbl> 0.906292763, 0.906292763, 0.318981223, 0.318981223…
    ## $ subeffect_okyrj     <dbl> 0.39198097, 0.39198097, 1.66331545, 1.66331545, 1.…
    ## $ subeffect_orbnj     <dbl> 1.4720569, 1.4720569, 1.8828733, 1.8828733, 1.8828…
    ## $ subeffect_sqgvz     <dbl> 0.82689639, 0.82689639, 0.22986343, 0.22986343, 0.…
    ## $ subeffect_uxabv     <dbl> 0.10700223, 0.10700223, 0.36757084, 0.36757084, 0.…
    ## $ subeffect_xqsts     <dbl> 0.19979161, 0.19979161, 0.62221192, 0.62221192, 0.…
    ## $ subeffect_xywyv     <dbl> 0.01874079, 0.01874079, 0.17953925, 0.17953925, 0.…

Then we do some plots

``` r
p1 <- drawsDF %>% ggplot(aes(x = condeffect_control)) +
  geom_histogram(bins = 250) +
  theme_minimal() +
  coord_cartesian(xlim = c(0, 0.5))
p2 <- drawsDF %>% ggplot(aes(x = condeffect_treatmnt)) +
  geom_histogram(bins = 250) +
  theme_minimal() +
  coord_cartesian(xlim = c(0, 0.5))
p1 / p2
```

![Figure
21](greta_causact_exploration_files/figure-gfm/unnamed-chunk-34-1.png)

## Model 8: lm(outcome\~condition+time)

Recall earlier that we noted that the outcome does not appear to be
linearly related to time, and so in the previous models we treated time
as a qualitative variable. Her we will select a subset of the data where
treating time as numeric *may* be reasonable, and show how to use that
in a model.

``` r
df %>%
  filter(between(time,8,16))%>%
  ggplot(aes(x = time, y = outcome)) +
  geom_point() +
  geom_smooth(method='lm', se=FALSE)+
  theme_minimal() +
  facet_wrap(covariate ~ condition) +
  theme(legend.position = "none")
```

![Figure
22](greta_causact_exploration_files/figure-gfm/unnamed-chunk-35-1.png)

``` r
df_t <- df %>% filter(between(time,8,16))
```

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df_t$outcome) %>%
  dag_node("mu", "mu", rhs = int + timeeffect*t, child = "o") %>%
  dag_node("intercept", "int",rhs=normal(0,10) , child = "mu") %>%  
  dag_node("time", "t", data = df_t$time, child = "mu") %>%
  dag_node("timeeffect", "timeeffect", rhs = uniform(-5, 5), child = "mu") %>%
  dag_plate("Observations", "i", nodeLabels = c("o","mu","t")) 
graph %>% dag_render()
```

![Figure
24](greta_causact_exploration_files/figure-gfm/unnamed-chunk-37-1.png)

``` r
drawsDF <- graph %>% dag_greta()
drawsDF %>% dagp_plot()
```

![Figure
24](greta_causact_exploration_files/figure-gfm/unnamed-chunk-37-2.png)

## Model 9: lm(outcome\~condition+time)

``` r
graph <- dag_create() %>%
  dag_node("outcome", "o", rhs = normal(mu, 3), data = df_t$outcome) %>%
  dag_node("mu", "mu", rhs = condeffect + timeeffect*t, child = "o") %>%
  dag_node("condeffect", "condeffect", rhs = uniform(0, 5), child = "mu") %>%
  dag_node("time", "t", data = df_t$time, child = "mu") %>%
  dag_node("timeeffect", "timeeffect", rhs = uniform(-5, 5), child = "mu") %>%
  dag_plate("Condition Effect", "k", nodeLabels = c("condeffect"), data = df_t$condition, addDataNode = TRUE) %>%
  dag_plate("Observations", "i", nodeLabels = c("o","mu","t")) 
graph %>% dag_render()
```

![Figure
25](greta_causact_exploration_files/figure-gfm/unnamed-chunk-38-1.png)

``` r
drawsDF <- graph %>% dag_greta()
drawsDF %>% dagp_plot()
```

![Figure
25](greta_causact_exploration_files/figure-gfm/unnamed-chunk-38-2.png)

# To Do

Further posts will talk about how to use greta and causact to
understand:  
- Multi-Level Models  
- Priors  
- Posterior Predictive Distributions

# Conclusion

There *may* (or many not - I am just starting with greta & causact) be
some things that Stan can do that greta and causact can not, but the
ease of creating models with greta and causact makes them a great place
to start, even if you eventually need more sophisticated tools.

# Appendices

<details>
<summary>

References

</summary>

[greta](https://greta-stats.org/index.html)  
[causact](https://www.causact.com/)  
[Stan](https://mc-stan.org/)  
[Rethinking](https://xcelab.net/rm/statistical-rethinking/)  
[RAOS](https://avehtari.github.io/ROS-Examples/index.html)  
[Workflow
1](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf)  
[Workflow
2](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)

</details>
<details>
<summary>

SessionInfo

</summary>

``` r
sessionInfo()
```

    ## R version 4.1.2 (2021-11-01)
    ## Platform: x86_64-apple-darwin17.0 (64-bit)
    ## Running under: macOS Big Sur 10.16
    ## 
    ## Matrix products: default
    ## BLAS:   /Library/Frameworks/R.framework/Versions/4.1/Resources/lib/libRblas.0.dylib
    ## LAPACK: /Library/Frameworks/R.framework/Versions/4.1/Resources/lib/libRlapack.dylib
    ## 
    ## locale:
    ## [1] en_CA.UTF-8/en_CA.UTF-8/en_CA.UTF-8/C/en_CA.UTF-8/en_CA.UTF-8
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ##  [1] causact_0.4.0   greta_0.3.1     bayesplot_1.8.1 lme4_1.1-27.1  
    ##  [5] Matrix_1.3-4    broom_0.7.10    janitor_2.1.0   DT_0.20        
    ##  [9] knitr_1.36      ggridges_0.5.3  patchwork_1.1.1 skimr_2.1.3    
    ## [13] lubridate_1.8.0 tictoc_1.0.1    forcats_0.5.1   stringr_1.4.0  
    ## [17] dplyr_1.0.7     purrr_0.3.4     readr_2.1.1     tidyr_1.1.4    
    ## [21] tibble_3.1.6    ggplot2_3.3.5   tidyverse_1.3.1
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] minqa_1.2.4        colorspace_2.0-2   ellipsis_0.3.2     rprojroot_2.0.2   
    ##  [5] snakecase_0.11.0   base64enc_0.1-3    fs_1.5.1           rstudioapi_0.13   
    ##  [9] listenv_0.8.0      farver_2.1.0       fansi_0.5.0        xml2_1.3.3        
    ## [13] codetools_0.2-18   splines_4.1.2      jsonlite_1.7.2     nloptr_1.2.2.3    
    ## [17] dbplyr_2.1.1       png_0.1-7          tfruns_1.5.0       DiagrammeR_1.0.6.1
    ## [21] compiler_4.1.2     httr_1.4.2         backports_1.4.0    assertthat_0.2.1  
    ## [25] fastmap_1.1.0      cli_3.1.0          visNetwork_2.1.0   htmltools_0.5.2   
    ## [29] prettyunits_1.1.1  tools_4.1.2        igraph_1.2.9       coda_0.19-4       
    ## [33] gtable_0.3.0       glue_1.5.1         reshape2_1.4.4     Rcpp_1.0.7        
    ## [37] cellranger_1.1.0   vctrs_0.3.8        nlme_3.1-153       xfun_0.28         
    ## [41] globals_0.14.0     ps_1.6.0           rvest_1.0.2        lifecycle_1.0.1   
    ## [45] future_1.23.0      MASS_7.3-54        scales_1.1.1       hms_1.1.1         
    ## [49] parallel_4.1.2     RColorBrewer_1.1-2 yaml_2.2.1         reticulate_1.22   
    ## [53] stringi_1.7.6      highr_0.9          tensorflow_2.6.0   boot_1.3-28       
    ## [57] repr_1.1.3         rlang_0.4.12       pkgconfig_2.0.3    evaluate_0.14     
    ## [61] lattice_0.20-45    htmlwidgets_1.5.4  labeling_0.4.2     cowplot_1.1.1     
    ## [65] tidyselect_1.1.1   processx_3.5.2     here_1.0.1         parallelly_1.29.0 
    ## [69] plyr_1.8.6         magrittr_2.0.1     R6_2.5.1           generics_0.1.1    
    ## [73] DBI_1.1.1          mgcv_1.8-38        pillar_1.6.4       haven_2.4.3       
    ## [77] whisker_0.4        withr_2.4.3        modelr_0.1.8       crayon_1.4.2      
    ## [81] utf8_1.2.2         tzdb_0.2.0         rmarkdown_2.11     progress_1.2.2    
    ## [85] grid_4.1.2         readxl_1.3.1       callr_3.7.0        reprex_2.0.1      
    ## [89] digest_0.6.29      webshot_0.5.2      munsell_0.5.0

</details>

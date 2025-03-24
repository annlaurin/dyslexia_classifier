rm(list=ls())

library(tidyverse)
df <- read.csv("hyperparameter_selection_full.csv", sep = ";")%>% 
  separate(X.Parameters, 
           into = c('lr','layers', 'dropout', 
                    'mask_prob', 'upscale_dim', 
                    'batch_size', 'num_heads', 
                    'inner_dim'), 
           sep = ",") %>%
  mutate(lr = gsub("\\{\\'lr\\': ", "", lr),
         layers = gsub("\\'num_layers\\': ", "", layers),
         dropout = gsub("\\'dropout\\': ", "", dropout),
         mask_prob = gsub("\\'mask_prob\\': ", "", mask_prob),
         upscale_dim = gsub("\\'upscale_dim\\': ", "", upscale_dim),
         batch_size =  gsub("\\'batch_size\\': ", "", batch_size),
         num_heads =  gsub("\\'num_heads\\': ", "", num_heads),
         inner_dim =  gsub("\\'inner_dim\\': ", "", inner_dim),
         inner_dim = gsub("\\}", "", inner_dim)
  )

minima <- df %>%
  filter(Precise_loss < 0.0001)

# what are the best params?
minima <- minima %>%
  select(-X.Tune.loss) %>%
  mutate_if(is.character, as.numeric)

minima %>% count(layers)
layer_plot <- minima %>% 
  mutate(layers = as.factor(layers),
         layers = fct_expand(layers,"16")) %>%
  ggplot() +
  scale_x_discrete(name = "Layers", drop = FALSE) +
  geom_bar(aes(x = layers), stat = "Count") + 
  theme_classic(base_size = 25) + 
  ylab("Count") +
  ggtitle("Number of encoder layers")
layer_plot
ggsave("img/layer_plot.pdf", width = 8, height = 8)

minima %>% count(lr)
lr_plot <- minima %>% 
  count(lr) %>%
  ggplot(aes(x=lr, y=n)) +
  geom_bar(stat="identity") +
  theme_classic(base_size = 25) + 
  ylab("Count") +
  xlab("Learning rate") +
  ggtitle("Learning rate")
lr_plot
ggsave("img/lr_plot.pdf", width = 8, height = 8)

minima %>% count(dropout)
dropout_plot <- minima %>% 
  count(dropout) %>%
  ggplot(aes(x=dropout, y=n)) +
  geom_bar(stat="identity") +
  theme_classic(base_size = 25) + 
  ylab("Count") +
  xlab("Dropout probability") +
  ggtitle("Dropout rate")
dropout_plot
ggsave("img/dropout_plot.pdf", width = 8, height = 8)

minima %>% count(mask_prob)
mask_plot <- minima %>% 
  count(mask_prob) %>%
  ggplot(aes(x=mask_prob, y=n)) +
  geom_bar(stat="identity") +
  theme_classic(base_size = 25) + 
  ylab("Count") +
  xlab("Masking probability") +
  ggtitle("Probability of masking")
mask_plot
ggsave("img/mask_plot.pdf", width = 8, height = 8)

minima %>% count(upscale_dim)
upscale_plot <- minima %>% 
  mutate(upscale_dim = as.factor(upscale_dim),
         upscale_dim = fct_expand(upscale_dim, "14", "512", "1024"),
         upscale_dim = factor(upscale_dim, levels(upscale_dim)[c(5,1,2,3,4,6,7)])) %>%
  ggplot() +
  scale_x_discrete(name = "Dimensions", drop = FALSE) + 
  geom_bar(aes(x = upscale_dim), stat = "Count") + 
  theme_classic(base_size = 25) + 
  ylab("Count") +
  ggtitle("Upscale dimensions") 
upscale_plot
ggsave("img/upscale_plot.pdf", width = 8, height = 8)

minima %>% count(num_heads)
heads_plot <- minima %>% 
  mutate(num_heads = as.factor(num_heads),
         num_heads = fct_expand(num_heads, "256")) %>%
  ggplot() +
  scale_x_discrete(name = "Heads", drop = FALSE) +
  geom_bar(aes(x = num_heads), stat = "Count") + 
  theme_classic(base_size = 25) + 
  ylab("Count") +
  xlab("Heads") +
  ggtitle("Number of heads")
heads_plot
ggsave("img/heads_plot.pdf", width = 8, height = 8)

minima %>% count(inner_dim)
inner_dim_plot <- minima %>% 
  mutate(inner_dim = as.factor(inner_dim),
         inner_dim = fct_expand(inner_dim, "2048")) %>%
  ggplot() +
  scale_x_discrete(name = "Inner dimensions", drop = FALSE) +
  geom_bar(aes(x = inner_dim), stat = "Count") + 
  theme_classic(base_size = 25) + 
  ylab("Count") +
  ggtitle("Inner dimensions")
inner_dim_plot
ggsave("img/inner_dim_plot.pdf", width = 8, height = 8)

#########################

minima %>% 
  count(layers, lr, dropout, mask_prob, 
        upscale_dim, batch_size, num_heads, inner_dim)



minima %>% 
  arrange(Precise_loss) %>%
  slice(1:5)


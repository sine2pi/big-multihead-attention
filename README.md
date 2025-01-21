##### Attentions scores are there and flowing but totally optional as both qk and relative positional bias have been rolled into the out tensor. 
##### It's there in case someone needs it for something. these are experiments...
##### Otherwise, the out is your standard shape [batch, seq_length, dims] tensor. 
##### Just be aware of the bias. It changes and adjusts to loss. The max can be set when you initialize your model via config hyperparameter nax_dist. 
##### Or leave it be, eventually it will find the right spot for itself. I would still keep an eyeball on it though. The changes in bias print to screen while training.
##### max_dist is your starting point for bias with a 1:1 token ratio.

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial


'''
This is extended version of the code from Jesse Vig(2019)
https://github.com/jessevig/bertviz
'''

def format_attention(attention):
   squeezed = []
   for layer_attention in attention:
      if len(layer_attention.shape) != 4:
         raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set output_attentions= True when initializing your model.")
      squeezed.append(layer_attention.squeeze(0))
   return torch.stack(squeezed)
# num_layers x num_heads x seq_len x seq_len

def format_special_chars(tokens):
   return [t.replace('Ġ', '').replace('▁', ' ').replace('</w>', '') for t in tokens]

class attention():
   def __init__(self,text,model,tokenizer):
      self.text = text
      self.model = model
      self.tokenizer = tokenizer

      inputs = self.tokenizer.encode_plus(text, return_tensors = 'pt', add_special_tokens = False)
      input_ids = inputs['input_ids']
      attention = model(input_ids)[-1]
      input_ids_list = input_ids[0].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
      self.tokens = format_special_chars(tokens)
      attn = format_attention(attention)
      
      self.attn_data = {
         "all":{
            "attn":attn.tolist(),
            "left_text":self.tokens,
            "right_text":self.tokens
            }
         }
      self.attn = self.attn_data["all"]["attn"]

   def attn_to_dataframe(self):
      word_i = [i+'_i' for i in self.tokens]
      word_j = [i+'_j' for i in self.tokens]
      df = pd.DataFrame(index = word_i,columns = word_j,dtype=float)
      for word_i in range(len(self.tokens)):
         for word_j in range(len(self.tokens)):
            i_to_j = []
            for layer in range(len(self.attn)):
               for head in range(len(self.attn[0])):
                  i_to_j.append(self.attn[layer][head][word_i][word_j])
            df.iloc[word_i].iloc[word_j] = np.mean(i_to_j)
      return df

   def show_plot(self,df):
      sns.heatmap(df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
      plt.show()
                                                          
                        
   def get_head_attention(self,layer,head):
      #get attention by tokens in a specific head
      attn_set = self.attn[layer][head]
      this_head_frame = pd.DataFrame(attn_set,index = self.tokens, columns = self.tokens)
      return this_head_frame
      


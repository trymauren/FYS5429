import numpy as np
import jax.numpy as jnp

def convert_txt_int(self, text) -> jnp.ndarray:
    with open(text, "r") as text:
       ascii_list = jnp.array(list(text))

    int_list = jnp.array()
    for ascii in ascii_list:
        int_list.append(_ascii_to_int(ascii))
    
    return int_list

def convert_int_txt(self,unicode_list):
    ascii_list = jnp.array()
    for unicode_int in unicode_list:
        ascii_list.append(_int_to_ascii(unicode_int))
    
    return ascii_list #Write this to an actual txt file, or just have as list to show predictions?

def _ascii_to_int(self,ascii):
    return ord(ascii)

def _int_to_ascii(self,int):
    return chr(int)
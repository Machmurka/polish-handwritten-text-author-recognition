from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pdb
from pathlib import Path
# print(Path(__file__).resolve())
file_desc_ptr = open('Data/author1/word_places.txt', 'r')
text = file_desc_ptr.read()
lines = text.split('\n')
image_file_name_prev=''
row_values = lines[2].split()
image_file_name = "Data/author1\\" + row_values[0][1:-1]
print(image_file_name,row_values)
if image_file_name != image_file_name_prev:   
    image = mpimg.imread(str(image_file_name))
    image_file_name_prev = image_file_name

plt.imshow(image)

plt.show()
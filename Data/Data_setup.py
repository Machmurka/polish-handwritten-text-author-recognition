import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pdb
from pathlib import Path
from collections import Counter
# print(Path(__file__).resolve())

class TestingSave:
    def __init__(self) -> None:
        pass

    def count_word_occurrences(self,AuthorText:str)->dict:
        '''
        Zliczenie poszczególnych słów dla pojedyńczego authora 
        Wykorzystywane dla zapisywanie słów jako png

        Args: 
            author_nr: Numer authora licząc od 1 

        Returns:
            dict ze słowem(str) i liczbą wystapienia w tekście(int)
            
            {'na': 46, 'w': 41,...}
        '''
        
        words=[]
        text = AuthorText
        lines = text.split('\n')
        number_of_lines = lines.__len__() - 1

        for i in range(number_of_lines):
            row_values = lines[i].split()
            if len(row_values) != 0 and row_values[0] != '%':
                words.append(row_values[1])
                # print(f"i: {i} row_values: {row_values[1]}")

        word_counts=Counter(words)

        
        return word_counts

    def save_word_to_png(subimage, filename)->None:
        '''
        Zapis słowa do pliku png w folderze autor/slowa 

        Arg:
            subimage: wycinek słowa z 
        '''
        if os.path.exists(filename):
            print(f"File {filename} already exists.")
        else:
            plt.imshow(subimage, cmap='gray')
            plt.axis('off') 
            plt.savefig(filename, bbox_inches='tight', pad_inches = 0)

        

    def save_words_to_file(self,Num_of_authors:int):
        ''''
            Zapisywanie w poczególnych folderach autora słów w formie BMP/PNG do przedyskutowania 
            PATRZ: https://www.adobe.com/pl/creativecloud/file-types/image/comparison/bmp-vs-png.html  
            
            Nazwa pliku : słowo + wystąpienie słowa

            Arg:
                Num_of_authors: Liczba autorów licząc od 0 dla których słową będą zapisane 

            
        '''
        num_of_authors=Num_of_authors
        for author_no in range(num_of_authors):
            file_desc_name = "Data/author" + str(author_no + 1) + "/word_places.txt"
            file_desc_ptr = open(file_desc_name, 'r')
            text = file_desc_ptr.read()
            lines = text.split('\n')
            number_of_lines = lines.__len__() - 1
            row_values = lines[0].split()
            number_of_values = row_values.__len__()

            image_file_name_prev = ""
            for i in range(number_of_lines):
                row_values = lines[i].split()

                if len(row_values) != 0 and row_values[0] != '%':
                    number_of_values = len(row_values)
                    image_file_name = "Data/author" + str(author_no + 1) + "\\" + row_values[0][1:-1]

                    if image_file_name != image_file_name_prev:   
                        image = mpimg.imread(str(image_file_name))
                        image_file_name_prev = image_file_name
                    word = row_values[1]

  
                    row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                        int(row_values[4]), int(row_values[5])
                    subimage = image[row1:row2,column1:column2] 
                    plt.title("Author "+str(author_no+1)+", image = "+row_values[0][1:-1]+", word = "+word)
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.imshow(subimage)
                    plt.show()

                    

        file_desc_ptr.close()
        
        

            



if __name__=='__main__':
    t=TestingSave()


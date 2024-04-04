import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import pdb
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
# print(Path(__file__).resolve())

class RawData:
    '''
        Przekształcanie poszczególnych słów autora z tekstówa na 
    '''
    def __init__(self) -> None:
        pass

    def count_word_occurrences(self,AuthorText:str)->dict:
        '''
        Zliczenie poszczególnych słów w tekście dla pojedyńczego authora 

        Args: 
            AuthorText: text autora (str) po użyciu funkcji .open() .read()

        Returns:
            dict ze słowem(str) i liczbą wystapienia w tekście(int)
            
            {'ala': 1, 'w': 41,...}
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
    
    def filter_words(self, AuthorText: str) -> dict:
        '''
        Zliczenie poszczególnych słów w tekście które występują więcej niż 1 raz dla pojedyńczego authora 


        Args: 
            AuthorText: text autora (str) po użyciu funkcji .open() .read()

        Returns:
            dict ze słowem(str) i liczbą wystapienia w tekście(int) >1 
            
            {'na': 46, 'w': 41,...}
        '''

        word_counts = self.count_word_occurrences(AuthorText)
        return {word: count for word, count in word_counts.items() if count > 1}
    
    def save_word_to_png(self, subimage, filename)->None:
        '''
        Zapis słowa do pliku png w folderze autor/slowa 
        Sprawdza czy plik o danej nazwie już nie istnieje dla danego autora

        Arg:
            subimage: wycinek słowa ze zdjęcia wykorzystując image[row1:row2,column1:column2]
            filename: Nazwa pod jaką słowo ma być zapisane

        '''
        if os.path.exists(filename):
            print(f"File {filename} already exists.")
        else:
            subimage = np.array(subimage)
            plt.imshow(subimage, cmap='gray')
            plt.axis('off') 
            plt.savefig(filename, bbox_inches='tight', pad_inches = 0)

        

    def save_words_to_file(self,Num_of_authors:int):
        ''''
            Zapisywanie w poczególnych folderach autora słów w formie (BMP/PNG do przedyskutowania )!!!
            PATRZ: https://www.adobe.com/pl/creativecloud/file-types/image/comparison/bmp-vs-png.html  
            
            Nazwa pliku : słowo + wystąpienie słowa w tekście ( Użycie count_word_occurrences() w celu ówczesnego zliczenia wystąpienia słowa )

            Arg:
                Num_of_authors: Liczba autorów licząc od 0 dla których słową będą zapisane 

            
        '''
        
        # System odczytu pliku zostaje taki sam jak w gotowym pliku word_display.py
        num_of_authors=Num_of_authors

        for author_no in range(num_of_authors):

            file_desc_name = "Data/author" + str(author_no + 1) + "/word_places.txt"
            file_desc_ptr = open(file_desc_name, 'r')
            text = file_desc_ptr.read()
            lines = text.split('\n')
            number_of_lines = lines.__len__() - 1
            row_values = lines[0].split()
            # number_of_values = row_values.__len__()
            
            # Będziemy potrzebować słów które występują więcej niż 1 raz 
            words=self.filter_words(text)

            image_file_name_prev = ""
            for i in tqdm(range(number_of_lines)):
                row_values = lines[i].split()

                if len(row_values) != 0 and row_values[0] != '%':
                    # number_of_values = len(row_values)
                    image_file_name = "Data/author" + str(author_no + 1) + "\\" + row_values[0][1:-1]

                    if image_file_name != image_file_name_prev:   
                        image = mpimg.imread(str(image_file_name))
                        image_file_name_prev = image_file_name



                    # Zmiana nazwy na słowo+wystąpienie 
                    word = row_values[1]
                    if word in words.keys():
                        words[word]-=1
                        word= word+str(words[word]) if words[word] > 0 else word
                        
                        # print(word)
                        # print(f"\nbefore\nword: {word}, count: {words[word]}")
                        # print(f"after \n word: {word}, count: {words[word]}")

                    filename="Data/author" + str(author_no + 1) + "/skany/slowa/"+word
                    if os.path.exists(filename) == False :

                        row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                            int(row_values[4]), int(row_values[5])
                        subimage = image[row1:row2,column1:column2] 


                        self.save_word_to_png(subimage,filename)
                    # plt.title("Author "+str(author_no+1)+", image = "+row_values[0][1:-1]+", word = "+word)
                    # plt.xlabel("X")
                    # plt.ylabel("Y")
                    # plt.imshow(subimage)
                    # plt.show()


                    

        file_desc_ptr.close()
        
        

            



if __name__=='__main__':
    t=RawData()
    t.save_words_to_file(1)

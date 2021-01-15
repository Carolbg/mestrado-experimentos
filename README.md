# Infrared termography and convolutional neural networks


## Entry points
The code has some entry points:

- Entire flow in python
    - scripts 
- Start flow in matlab for preprocessing and run python after
    - scripts 
- AGs


## Matlab Codes:
Some matlab code are the first step and should be run before running python code due to the fact that they generate the data the python might use.

- minMax_preprocessing -> Realiza o pre processamento utilizando o min max no matlab e salva os numpy array. 
    - Pode ou não chamar a geração de aumento de dados
- colormap_preprocessing.m -> Realiza o pre processamento utilizando o jet colormap  no matlab e salva os numpy array. 
    - Pode ou não chamar a geração de aumento de dados
    - Não usa o min max nem ratio já que o range dos valores com a conversão do jet resulta em valores entre 0 e 1
- colormap_readingAllFiles -> Realiza o pre processamento utilizando o jet colormap  no matlab e salva os numpy array. 
    - A diferença desse para o de cima é que ele lê todos os arquivos da pasta enquanto o de cima lê os arquivos com nome especificados no array.
    - Pode ou não chamar a geração de aumento de dados
    - Não usa o min max nem ratio já que o range dos valores com a conversão do jet resulta em valores entre 0 e 1
- ratio_preprocessing

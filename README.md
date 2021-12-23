#Convolucões em Processamento Digital de Imagens
### Autor: [Jaylton Alencar Pereira](https://github.com/jaylton)
#### Trainee - Inteligência Artifical @ Asa Branca Aerospace

## Introdução





Na matemática convolução é uma operação de somatório do produto entre duas funções, ao longo da região em que elas se sobrepõem, em razão do deslocamento existente entre elas. Na teoria, o cálculo da convolução contínua é dada pela integral deste produto, desde que as funções sejam integráveis no intervalo:

<div align="center">
<img src="https://latex.codecogs.com/gif.latex?\LARGE&space;(f*g)(x)&space;=&space;h(x)&space;=&space;\int_{-\infty&space;}^{\infty&space;}&space;f(u)*g(x-u)du" title="\LARGE (f*g)(x) = h(x) = \int_{-\infty }^{\infty } f(u)*g(x-u)du" />
</div>

Na computação, utiliza-se o cálculo discreto, que realiza um somatório do produto:

<div align="center">
<img src="https://latex.codecogs.com/gif.latex?\LARGE&space;(f*g)(k)&space;=&space;h(k)&space;=&space;\sum_{j=0}^{k}&space;f(j)*g(k-j)" title="\LARGE (f*g)(k) = h(k) = \sum_{j=0}^{k} f(j)*g(k-j)" />
</div>

Quando se trata de utilizar a convolução em processamento de imagens para Inteligência Artificial, são necessários dois somatórios pois temos duas dimensões – altura e largura:

<div align="center">
<img src="https://latex.codecogs.com/gif.latex?\large&space;(f*g)(x,y)&space;=&space;\sum_{i=-\infty&space;}^{\infty&space;}&space;\sum_{j=-\infty&space;}^{\infty&space;}&space;f(i,j)*g(x-i,y-j)" title="\large (f*g)(x,y) = \sum_{i=-\infty }^{\infty } \sum_{j=-\infty }^{\infty } f(i,j)*g(x-i,y-j)" />
</div>

As convoluções em processamento de imagens tem como principal objetivo aplicar filtros para extração de informações de interesse nas imagens, filtros que podem ser de vários tipos, como filtros de contornos, filtros passa-baixas,  filtros passa-altas e filtros detectores de bordas, cada um desses filtros será melhor detalhado logo abaixo. O funcionamento da convolução é dado da seguinte forma, uma imagem pode ser representada por uma matriz, essa matriz é passada por um kernel (núcleo) e o resultado será a imagem filtrada, o kernel faria de acordo com aquilo que se deseja fazer. Uma demostração disso pode ser vista na seguinte imagem.

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1X-uww6m3DioZZO_mhThNcU-0F0u5JiyW">
</p>
Neste exemplo o kernel usado tem as deminsões definidas por (5,5,3), mas isso dependo daquilo que se deseja fazer com o filtro e também da imagem filtrada.

## Filtro passa-baixa
Esta é a mais básica das operações de filtragem. Um filtro passa-baixa, também chamado de filtro de desfoque ou suavização, calcula a média das mudanças rápidas de intensidade. O filtro passa-baixa por média, calcula apenas a média de um pixel e todos os seus oito vizinhos imediatos, o resultado substitui o valor original do pixel e o processo é repetido para cada pixel da imagem.


```python
img = cv2.imread("imgRuido.jpg", 0) #Imagem de entrada

blur = cv2.blur(img,(5,5)) #Imagem de saída

```

Imagem de entrada

<img src="https://drive.google.com/uc?export=view&id=1r-0IurNZUEHqO2tgm1yPCYlyguC2fArv">

Imagem de saída

<img src="https://drive.google.com/uc?export=view&id=1g7ioJb7Byd5j-1whj-uvqAwxV6WlACPd">


O filtro passa baixa tambem pode ser implementado por um filtro gaussiano.  O filtro gaussiano é altamente eficaz na remoção de ruído da imagem, ele analisa a vizinhança ao redor do pixel e encontra sua média ponderada gaussiana.


```python
img = cv2.imread("imgRuido.jpg", 0) #Imagem de entrada

blur = cv2.GaussianBlur(img,(5,5),0) #Imagem de saída

```

Imagem de entrada

<img src="https://drive.google.com/uc?export=view&id=1r-0IurNZUEHqO2tgm1yPCYlyguC2fArv">

Imagem de saída

<img src="https://drive.google.com/uc?export=view&id=1ZJ5ksT-06GN8b1C5nOrxO6dMw9PxtkJD">

O filtro bilateral também é um filtro do tipo passa baixa, esse filtro é altamente eficaz na remoção de ruído enquanto mantém as bordas afiadas. Mas a operação é mais lenta em comparação com outros filtros. Este filtro também é baseado no método gaussiano, mas ele é um filtro gaussiano que leva em consideração a diferença de intensidade dos pixels. Diferentemene do filtro gaussiano citado anteriormente, o filtro bilateral garante que apenas os pixels com intensidade semelhante ao pixel central sejam considerados para desfoque. Portanto, ele preserva as bordas, pois os pixels nas bordas terão grande variação de intensidade. 


```python
img = cv2.imread("imgRuido.jpg", 0) #Imagem de entrada

blur = cv2.bilateralFilter(img,5,75,75) #Imagem de saída

```

Imagem de entrada

<img src="https://drive.google.com/uc?export=view&id=1r-0IurNZUEHqO2tgm1yPCYlyguC2fArv">

Imagem de saída

<img src="https://drive.google.com/uc?export=view&id=1Ntam4NTEqP5qtFMHD28B_4UFEQe_D0jA">

##Filtro passa-alta
Um filtro passa-alta é um filtro que examina uma região de uma imagem e aumenta a intensidade de certos pixels com base na diferença de intensidade com os pixels estão ao redor. Este tipo de filtro ajuda a encontrar bordas nas imagens. 

Após calcular a soma das diferenças das intensidades do pixel central em relação a todos os vizinhos imediatos, a intensidade do pixel central será aumentada (ou não) se um alto nível de alterações for encontrado. Em outras palavras, se um pixel se destacar dos pixels ao redor, ele será aumentado.


```python
from scipy import ndimage

# Criando um Kernel de 5x5 
kernel_5x5 = np.array(
[[-1, -1, -1, -1, -1],
[-1, 1, 2, 1, -1],
[-1, 2, 4, 2, -1],
[-1, 1, 2, 1, -1],
[-1, -1, -1, -1, -1]])

# Imagem de entrada
img = cv2.imread("img_escudo.jpg", 0)

# Aplicando a convolução e obtendo a imagem de saída
new_img = ndimage.convolve(img, kernel_5x5)

```

Imagem de entrada

<img src="https://drive.google.com/uc?export=view&id=1GK9E3P5JFc8tKH8pdziKKxhGiZoh7E27">

Imagem de saída

<img src="https://drive.google.com/uc?export=view&id=1ySD948ZXRTPQPGL-hg2sOlN7almd1wUc">

## Algoritmo de Canny

O algoritmo de Canny é utilizado para detecção de bordas nas imagens, ele é bastante complexo, é um processo dividido em cinco etapas que analisa a imagem com um filtro Gaussiano, calcula gradientes, aplica supressão não máxima nas bordas, para minimizar as bordas, aplica um limite duplo em todas as bordas detectadas para eliminar falsos positivos e analisa todas as arestas e sua conexão entre si para manter as arestas reais e descartar as falsas.


```python
img = cv2.imread("img_escudo.jpg", 0) #Imagem de entrada

filt = cv2.Canny(img, 200, 300) #Aplicando o algoritmo de Canny

```

Imagem de entrada

<img src="https://drive.google.com/uc?export=view&id=1GK9E3P5JFc8tKH8pdziKKxhGiZoh7E27">

Imagem de saída

<img src="https://drive.google.com/uc?export=view&id=1PcfAKuAi5NdbpJXw4bwY_0eVcVtvIGRq">

## Detecção de contorno
Outra tarefa vital na visão por computador é a detecção de contornos, não apenas por causa do aspecto óbvio de detectar contornos de elementos contidos em uma imagem ou quadro de vídeo, mas por causa das operações derivadas conectadas com a identificação de contornos.

Essas operações são, nomeadamente, calcular polígonos delimitadores, aproximar formas e, geralmente, calcular regiões de interesse, o que simplifica consideravelmente a interação com dados de imagem porque uma região retangular com NumPy é facilmente definida com uma fatia de matriz. Essa técnica é muito usada para detecção de objetos.


```python
img = cv2.imread("img_escudo.jpg", 0) #Imagem de entrada

# Obtendo thresholding entre 127 e 255 de intensidade
ret, thresh = cv2.threshold(img, 127, 255, 0)

# Obtendo o contorno da imagem
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#Desenhando as bordas verdes e obtendo o resultado
img = cv2.drawContours(color, contours, -1, (0,255,0), 4)

```

Imagem de entrada

<img src="https://drive.google.com/uc?export=view&id=1GK9E3P5JFc8tKH8pdziKKxhGiZoh7E27">

Imagem de saída

<img src="https://drive.google.com/uc?export=view&id=1bx6GEPhocF6-MoNPLKWRX7URF6i9rlsS">

## Filtro Laplaciano
Assim como o algortimo de Canny o filtro laplaciano pode ser usado para detectar as bordas de uma imagem. O Laplaciano de uma imagem destaca as áreas de mudanças rápidas de intensidade e, portanto, pode ser usado para detecção de bordas. Definindo a intensidade da imagem como I(x, y), o Laplaciano da imagem é dado pela seguinte equação: 

A aproximação discreta do Laplaciano em um pixel específico pode ser determinada tomando a média ponderada das intensidades do pixel em uma pequena vizinhança do pixel. 


```python
img = cv2.imread("pessoa.jpg", 0) #Imagem de entrada

new_img = cv2.Laplacian(img,cv2.CV_64F) #Aplicando o algoritmo de Canny

```

Imagem de entrada

<img src="https://drive.google.com/uc?export=view&id=1i7q5TYH2lCKD4FZ7NUjgG_cio2Ar0dMD">

Imagem de saída

<img src="https://drive.google.com/uc?export=view&id=1Eqf11T0bLRtDAaZdg_7A9Ze-WRgXNA-i">

## Conclusão
Filtros são muito importantes para o processamento de imagens, filtros de detecção de bordas podem ser usados para identificar padrões em imagens e assim identificar um objeto presente na imagem, filtros passa-baixa podem ser usados para melhorar a qualidade de uma imagem e  filtros passa-alta podem ser usados para obter caracteristicas especificas das imagens. Por fim, além de ser importatnes, todos esses filtros podem ser aplicados e testados facilmente com a biblioteca OpenCV.

### Fontes

https://viceri.com.br/insights/entendendo-de-vez-a-convolucao-base-para-processamento-de-imagens/

https://rsdharra.com/blog/lesson/9.html

https://rsdharra.com/blog/lesson/8.html

https://rsdharra.com/blog/lesson/11.html

https://cdn.diffractionlimited.com/help/maximdl/Low-Pass_Filtering.htm

https://towardsdatascience.com/image-filters-in-python-26ee938e57d2

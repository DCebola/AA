## Minimum Squared Error in Regression ##

> Em problemas de regressão pretendende-se, dado um set X de exemplos, uma função F(X) desconhecida e um set de labels Y contínuo, descobrir um modelo. O modelo é a função com um set de parâmetros T,  g(T, X) : X -> Y, que melhor se aproxima de F(X) : X -> Y.  Consideremos todas as hipóteses geradas como funções do tipo y = F(x) + erro.  Assumindo que o erro é aleatório e com distribuição normal, é possível estimar a likelihood das hipóteses. Para escolher a melhor hipótese é necessário encontrar a hipótese com a likelihood máxima. Simplificando, pode-se procurar antes pela hipótese que maximiza o logaritmo da likelihood, uma vez que é a mesma que maximiza a likelihood. Considerando que o sampling de valores da hipótese é também aleatório e com distribuição normal, encontrar a hipótese que maximiza o logaritmo da likelihood é equivalente a encontrar a hipótese com menor erro quadrático médio.

## Cross Validation - Erro de validação

> Na validação cruzada divide-se os dados de treinos em subsets mais pequenos, estes subsets são usados para gerar diferentes hipóteses. Para não estar a escolher uma hipótese através de uma heurística biased (por exemplo, através da seleção do menor erro quadrático médio), reserva-se um subset dos dados de treino para validar as hipóteses. O erro gerado entre as hipóteses e este set de validação é portanto apenas uma estimativa do erro da hipótese que será usado para escolha do modelo.

## Logistic Regression

> Regressão logística utiliza a função logistica como loss function para modelar a probabilidade de um exemplo pertencer a uma dada classe como função de um vetor de features. Para se maximizar esta probabilidade, recorrendo ao gradiente, minimiza-se a função logística. Assim é possível encontrar a maior likelihood e, por conseguinte, a melhor hipótese.

__[Not linearly separable]__ 

> Expandir as dimensões das features dos dados permite a utilização de discriminantes lineares e a separação dos exemplos
> em classes nestes feature spaces de maiores dimensões. O hiperplano que representa a boundary de decisão é depois novamente representado no
> feature space com as dimensões originais.

## Regularização

> Regularização é um método de reduzir a probabilidade de overfitting de um modelo, através da penalização da sua flexibilidade.

__[Linear regression]__

> No caso da regressão linear, para cada modelo polinomial, adiciona-se ao erro uma penalização em função do valor dos coeficientes do modelo.

__[Logistic regression]__

> No caso da regressão logística considera-se um parâmetro de regularização C - no caso de se estar a usar regularização do tipo L2 como é exemplo a ridge regression este toma valor 1/lambda. Este parametro será usado para controlar o grau de penalização a associar aos coeficientes do hiperplano gerado para separar as classes. Esta variação força o declive da função logistica a atenuar o seu declive assentuado, diminuindo a probabilidade de overfit.

__[Perceptrons]__

> No caso das redes neuronais, durante o treino, vão sendo selecionados exemplos de treino aleatórios como input, vai-se calculando o erro a cada época e usando cross-validation para detetar overfit. Mesmo que o erro ainda não tenha convergido, em caso de deteção de overfit, a paragem do processo evitará overfit da rede à custa da sua flexibilidade - ora esta paragem do treino pode ser considerado como um método de regularização.

__[SVM]__

> A margin classifier is a classifier that provides a measure of the distance between the frontier and the points closest to
> it. This is the margin of the classifier. A maximum margin classifier is a classifier that maximizes
> this distance. With logistic regression, we can approximate this using regularization, but this requires
> modifying the loss function to include the regularization term.

## Lazy vs Eager

> Em Eager Learning os dados de treino são usados para treinar um modelo e gerar uma hipótese à priori.
> Em Lazy Learning o processo de treino do modelo é adiado até ao momento em que se pretende obter um previsão. 

__[Naive Bayes]__

> Não se pode considerar o classificador como um exemplo de lazy learning porque se estão a usar, para fazer as previsões, os logaritmos já gerados das probabilidades condicionais dos atributos em cada classe. Isto é um exemplo de eager learning, onde modelos treinados geram hipóteses à priori para depois serem utilizadas para fazer previsões.

## Generative vs Discriminative

> Classificadores discriminantes prevêm a classe de um exemplo com base numa estimativa de uma probabilidade. Esta é a probabilidade condicional do ponto pertencer a uma dada classe, com base nas suas features.
> Classificadores geradores calculam as distribuições das probabilidades conjuntas das classes e features para depois gerarem previsões com base nestas distribuições. A partir destas distribuições é possível gerar novos valores de exemplo para cada classe.

## Classifiers comparison

__[Approximate Normal Test]__

> Assumindo que o número de erros é resultante da soma de variáveis aleatórias independentes, este número tende para uma distribuição normal
> com a sua média igual ao número espéctavel de erros. Pelo que:
> 	Erros(95%) = errs +- 1.96 * sqrt(mean(errs) * (1 - (errs/size_of_test_set)))
> 	Erros(95%) = errs +- 1.96 * std(errs)
> Calculando os intervalos de confiança do número de erros de dois classificadores se estes se intercetarem não é possível excluir a hipótese do seu true error ser igual, caso contrário o classificador com o menor valor é o melhor.
> Se o número de erros for bastante reduzido o teste não é muito fiável, pelo que se usam para valores de erros superiores a 5.

__[McNemar's Test]__

> No McNemar's test analisam-se os exemplos que cada classificador classificou corretamente quando o outro cometeu um erro.
> A divisão da diferença destes dois valores pela sua soma aproxima-se de uma distribuição qui-quadrado, com um grau de liberdade.
> Dadas estas propriedades, se o valor do rácio for superior a 3.84, é possível concluir com 95% de confiança que os classificadores têm
> performances diferentes.

## Perceptrons

__[Treino] __

> Treinar um percetrão faz-se minimizando o erro quadrático da resposta do percetrão com base na classe pretendida. 
> Calculando a derivada do erro como função dos pesos do percetrão é possível descobrir como aumentar os pesos do mesmo. Dado o encadeamento do erro ser função da ativação, desta ser função da soma pesada dos inputs e desta soma ser função dos pesos, é possível usar uma regra de encadeamento para calcular as derivadas destas funções. Obtem-se assim o gradiente como função de cada peso. 
> A partir disto consegue-se determinar a função de update para cada peso, dado um exemplo. A minimização do erro é feita depois através
> da descida ao longo do seu gradiente - pode ser feito em steps ou em batches para todo o training set (epoch).

__[MultiClass]__

> Adicionar na output layer mais nós

__[Linear Separability]__

> Adicionar uma hidden layer com um número de percetrões equivalente ao número de features. Estes percetrões irão contribuir com o seu
> output para a layer seguinte. Realizado o output da layer final é necessário calcular a soma dos erros obtidos e propagar estes 
> valores às camadas anteriores para reajuste dos pesos de todos os percetrões. Eventualmente os neurónios da hidden layer aprendem
> a transformar o training set num set linearmente separável. Este novo set, output da hidden layer e input das layers seguintes, já será
> capaz de ser separado pelos percetrões das layers finais.

## SVM

__[Not linearly separable]__

> Para classificar um set não linearmente separável será primeiro necessário adicionar um variável de folga por cada vetor, 
> que na realidade é um valor positivo que representa a distância entre o vetor e o interior da margem, ou zero se este não estiver 
> no interior da mesma. O processo de introdução destas folgas e de permitir a entrada de vetores nas margens é penalizado através 
> de uma variável de regularização para evitar overfit.
> Como se trata de um classificador linear será também necessário expandir o espaço de dimensões das
> features. No caso do Support Vector Machine isto é feito implicitamente através da utilização de kernel functions.

## Bias Vs Variancy

__[Bias]__

> Bias, no escopo de um ponto do modelo é a diferença entre o true value e o valor que se espera da previsão do modelo naquele ponto.
> No escopo do modelo inteiro é a média do bias em cada ponto. 

__[Variancy]__

> A variância, no escopo de um ponto do modelo é a variação esperada dos valores estimados para esse ponto. (assumindo que o modelo pode ser
> treinado por qualquer dataset).
> No escopo do modelo inteiro é a média das variâncias para todos os pontos.

__[Underfit]__

> Se um modelo está em underfit, o modelo não é capaz de se ajustar aos dados e a principal componente do erro é o bias.

__[Overfit]__

> Se um modelo está em overfit, o modelo ajustou-se demasiado aos dados de treino e é incapaz de generalizar.
> A principal componente do erro é a variância.
>
> Escolha ótima é o modelo que minimize tanto a variância como o bias.

##  Bootstrap 

> Para estimar o bias e a variância do modelo é necessário que o modelo tenha sido treinado sobre diversos training sets.
> Como normalmente só existe um training set, recorre-se a bootsrapping para gerar diversos subsets de treino através de um
> sampling aleatório sobre o training set. Como se calcula o erro para cada hipótese usando as réplicas que não foram usadas para 
> o seu treino, este calculo não é biased. Como os erros calculados para cada hipótese não são biased, o bias e a variância também não o serão.Reduz a variância sem  aumentar o bias. Útil quando o base model tem variância alta e bias reduzido.



## Ensemble 

> O objetivo de métodos ensemble é combinar diferentes modelos de modo a melhorar qualidade das previsões.
> Para isto pode-se treinar o modelo em várias réplicas dos dados obtidas por bootstrap e depois combinar as réplicas.
> O probabilidade das previsões melhorarem através da junção de vários classifiers assume que estes são estatísticamente independentes.
> Por isto é que é aconselhado usar métodos de ensemble sobre classificadores com variância alta, pois será com estes que o ganho
> da junção das hipóteses é o maior. Classificadores estáveis variam pouco pelo que as várias hipóteses terão previsões muito semelhantes,
> pelo que o ganho da sua junção será mínimo.

__[VARIANCE]__
	__[Bagging] __
		Em bagging a junção das hipóteses consiste em fazer a média de todas as previsões.
	__[Bragging]__
		Em bragging a junção das hipóteses consiste em fazer a mediana de todas as previsões.

__[BIAS]__
	__[AdaBoost]__
		Este método consiste em fazer a combinação linear de classificadores fracos. Nesta combinação linear a cada classificador
		é atribuído um peso. A soma dos pesos de cada classificador tem de ser 1 e têm de permitir que a soma da média ponderada 
		das estimativas do classificador resultante minimize o bias. Os pesos apenas podem diminuir.
		Isto é concretizado treinando cada classificador com os mesmos dados de treino mas dando pesos diferentes a cada exemplo.
		Os peso de exemplos mal classificados devem ser mais altos.  













# Learning Theory

### Empirical Risk Minimization 

> O risco empírico é o custo esperado aplicado ao set de treino. Minimização deste risco consiste na minimização do erro de treino, ou seja, na minimização de uma função de custo aplicada ao set de treino (e.g erros de classificação ou erro quadrático). No entanto, esta minimização não pode ser feita com o ajuste de parâmetros tendo por base o valor do risco empírico. Nesta situação a estimativa do risco absoluto (true risk), recorrendo ao risco empírico, seria biased.

### Probably Approximately Correct Learning

>Não é possível garantir que o true error seja zero mas  é possivel aumentar a probabilidade que este seja baixo, dado que existem dados suficientes. Normalmente, melhorar a melhor hipótese implica aumentar o tamanho do espaço de hipóteses, daí ser necessário o *inductive bias*
>
>E(ĥ) = ( arg h∊H min E(H) ) + 2 sqrt( (1 / 2m) * ln( |H| / δ ) )
>
>O primeiro termo é o menor true error de qualquer hipótese no espaço de hipotese H. Corresponde ao Bias do modelo e, quanto maior for este termo, menor será a capacidade do modelo se ajustar aos dados. Se for o termo dominante do true error, o modelo está numa situação de underfitting.
>
>O segundo termo é uma função entre o tamanho do espaço de hitpóteses e o tamanho do set de treino e corresponde à Variância do modelo. Quanto maior o espaço de hipóteses maior será a variância das previsões das hipóteses, obtidas pelo treino de diferentes sets de treino. Se for o termo dominante então o modela estará numa situação de overfitting, ajustando-se excessivamente aos dados do set de treino e perdendo capacidade de generalização.

### Shattering and the V-C Dimension (Vapnik-Chervonenkis dimension)

> No caso de classificadores com parâmetros contínuos não é possível assumir um número finito de hipóteses pelo que o espaço de hipóteses terá de ser infinito. Dado que passam a existir infinitas hipóteses que conseguem dividir os pontos corretamente deixa de ser relevante conhecer o seu número. Passa ser necessário conhecer o maior set de pontos que o classificador consegue dividir corretamente.
>
> A *V-C dimension* de H, VC(H), define-se como o tamanho do maior set de pontos que H consegue *shatter*.
>
> Um espaço de hipóteses H *shatters* um set de points S se, para cada label s ∊ S, existe uma hitpótese h ∊ H que é consistente com s. Mais concretamente, H *shatters* um set de pontos se existirem hipótese s que classificam corretamente todos os pontos, independentemente da sua classe.

# Unsupervised Learning #

> Processo de ajustar um modelo à estrutura dos dados sem recorrer ao uso de labels conhecidas, ou seja, sem recurso a uma medida de erro. O objetivo de aprendizagem não supervisionada é identificar aspetos dos dados (e.g a sua distribuição, relações entre features...).

## Feature Selection ##

> Nem todas as features dos dados têm a mesma utilidade e informação, por vezes, é necessário fazer a uma seleção das melhores, tanto para aprendizagem supervisionada como não supervisionada. Por exemplo, o número de features pode ser excessivo para os dados e  levar a overfit, algumas features poderão não ter informação relevante ou conter bastante ruído e certas features poderão ainda estar correlacionadas entre si.
Feature Selection é o processo de manter apenas as features adequadas, mantendo apenas as melhores. É possível fazer esta seleção antes do processo de aprendizagem, durante a aprendizagem ou depois, com base na performance das hipótese obtidas

### Filtering ###
> Se o processo de feature selection for executado antes do treino, denomina-se por filtering. 

### Univariate Filtering ###
> Se a filtragem tiver por base uma análise individual de cada feature trata-se univariate filtering. Neste tipo de filtering pode-se usar dados com labels e comparar cada feature à label do seu exemplo. Esta análise serve para determinar a relevância da feature para a previsão da label. Uma feature é relevante se se correlaciona com a label. Contrariamente, uma feature é irrelevante se for estatisticamente independente da label, caso em que é rejeitada.

#### Qui-Squared Test

>Um critério de seleção de features pode ser a sua independência estatística da label do seu exemplo, dado que features estatísticamente independentes da label são inúteis para a sua previsão. Isto pode ser calculado através do Chi-Squared test. Features com valor reduzido podem ser eliminadas pois estão perto de ser estatisticamente independentes da label. Só aplicável a features categóricas.

#### Analysis of Variance (ANOVA) F-Test

>O ANOVA F-Test compara a variância da features entre labels e dentro da própria label. Se o valor do F-Test é baixo a probabilidade da feature ser independente da label é alta pelo que pode ser rejeitada.

### Multivariate Filtering ###
> Se no processo de selecção de uma feature a análise tiver um escopo que involva mais do que uma feature trate-se de multivariate filtering. Neste tipo de filtering analisa-se o quão redundantes as features são em relação a outras features.

#### Correlation Analysis

>Para determinar a redundância de features pode-se analisar a correlação que esta têm entre si. Features com alta correlação são redundantes pois contribuem com a mesma informação. Dado um set de features com uma correlação elevada basta conservar uma.

### Wrapper Methods

>É possível fazer uma seleção de features com base na performance da hipótese treinada.  Este tipo de métodos denominam-se de wrapper methods e consistem em avaliar esta performance para diferentes subsets de features, de modo a determinar qual o melhor conjunto de features.

#### Sequential Forward Elimination

>Neste método começa-se com um set de features vazio e, a cada iteração avalia-se qual a melhor das restantes a adicionar ao set. O método é repetido até se obter o número de features ou a performance desejada.

#### Sequential Backward Elimination

>Neste método começa-se com um set com todas as features e, a cada iteração avalia-se qual a feature a eliminar de modo a maximizar a performance da hipótese. O método é repetido até se obter o número de features ou a performance desejada.

#### Non-Deterministic

>Wrapper methods determinísticos são greedy pelo que, com recursos limitados, utilizam-se wrapper methods não determinísticos. Nestes métodos, tenta-se maximizar a performance da hipótese com recurso a uma pesquisa de subsets de features através de algoritmos não determinísticos (e.g simulated annealing ou algoritmos genéticos).

### Embedded Feature Selection

>Certos algoritmos de aprendizagem incorporam feature selection. 
>
>É o caso de **árvores de decisão com profundidade limitada** onde as melhores features são usadas primeiro e, ao limitar-se a profundidade da árvore as peores features acabam por ser rejeitadas.
>
>Considerando **Naive Bayes** com pesos nas features. Se os pesos tiverem um valor proporcional à diferença entre a distribuição condicional dos valores de uma feature dada a classe e a distribuição da probabilidade à priori da classe, valorizam-se as melhores features e implicitamente rejeitam-se as piores.
>
>**Regularização** também pode levar a feature selection implicitamente, basta considerara regularização L1. Este tipo de regularização penaliza a soma dos valores absolutos das features, forçando algumas terem valores nulos e a serem rejeitadas.

## Feature Extraction

> Outro modo de reduzir as dimensões dados. Consiste na criação de novas features com base numa função sobre as features originais dos dados.

### Principal Component Analysis (PCA)

>PCA é um processo que uma transformação do set de dados num set ortogonal de coordenados  de modo a que os valores ao longo de cada coordenada são linearlimente não correlacionados. De um modo abstrato, o PCA escolhe a direção ao longo dos pontos de dados com a maior variância, e projeta os dados nesta direção, o principal componente. Iterativamente continua-se a fazer o mesmo sobre a projeção obtida, continuando-se a escolher direções que maximizem a variância e que sejam orthogonais a todas as escolhidas previamente. O processo executa até só restar uma direção orthogonal.
>
>Na prática não se implementa PCA neste modo de execução iterativo. Implementa-se sim, calculando os eigenvectors e respetivos eigenvalues, da matriz de covariância, ou mais precisamente, da scatter matrix. Dados o valor dos eigenvalues, podem-se se ordenar os eigenvectors e assim determinar os principal components por ordem de importância.

### Self Organizing Maps (SOM)

>Pode se imaginar um SOM como um rede neuronal artificial onde os neurónios têm um set de coeficientes com as mesmas dimensões dos dados e onde estes neurónios estão dispostos numa matriz bidimensional. Deste modo é possível calcular a distância entre os neurónios e os seus vizinhos, ou entre os neurónios e exemplos dos dados. 
>
>O treino de um SOM é iniciado com a atribuição de valores random aos coeficientes dos seus neurónios. Depois, iterativamente, descobre-se o neurónio mais próximo de um dado exemplo dos dados. Este neurónio denomina-se BMU (Best Matching Unit), e será transladado de modo a aproximar-se do exemplo dos dados. Os valores dos coeficientes dos restantes neurónios também são alterados segundo a mesma direção, no entanto, com uma escala gradualmente mais pequena, proporcional à distância ao BMU. A magnitude das mudanças é função do coeficiente de aprendizagem, que diminui monotonicamente durante o treino.

## Clustering

>O objetivo de clustering é agrupar exemplos similares e separar exemplos diferentes em diferentes grupos, clusters, segundo uma medida de similaridade. A formação dos grupos deve maximizar a similaridade entre exemplos do mesmo cluster e minimizar a similaridade entre exemplos de diferentes clusters.

### Prototype Based Clustering

> A atribuição de uma classe a um dado exemplo é feita com base na distância ao protótipo mais próximo. Só funciona bem em clusters globulares.

#### K-Means

> O algoritmo consiste em dividir os dados em k clusters, cada qual definido pelo ponto médio de todos os exemplos pertencentes ao cluster. Este vetor é o protótipo do cluster.
>
> O algoritmo inicia-se com a geração de um set random de k prototipos. De seguida atribui-se o exemplo mais próximo de cada protótipo ao cluster do protótipo. Volta-se a gerar os protótipos como o ponto médio de todos os exemplos dos clusters. Iterativamente repetem-se os dois últimos passo até se chegar a convergência ou o um critério de pausa.

#### K-Metoids

> Difere do K-Means porque todos os protótipos coincidem com exemplos. Deixa de ser necessário avaliar distâncias e pode-se só avaliar a similaridade entre os protótipos e os exemplos.
>
> O algoritmo inicia-se com inicialização dos k protótipos. De seguida, atribui-se os exemplos aos clusters. Para cada medoid e cada exemplo, testa-se se trocar o medoid pelo exemplo reduz a soma dos pares de dissimilaridades entre os exemplos e respetivos medoids. Em caso positivo, faz-se a troca. Repetem-se os dois últimos passos até não existirem melhorias. 

### Probabilistic Clustering 

> A atribuição de uma classe a um dado exemplo depende de uma probabilidade. Pode-se considerar um modelo probabilistico dos exemplos pertencentes a X e clusters pertencentes a Y, *P*(*X*,*Y*)=*P*(*X*|*Y*) *P*(*Y*).

#### Hidden Markov Models

> Para instantciar um modelo HMM com N estados é necessário: uma matriz de NxN onde N representam os estados do modelo; uma matriz NxK onde K representam os valores desses estados; as N probabilidades de valores de cada estado.

##### Forward algorithm

> A likelihood de sequências de observações. 

##### Viterbi algorithm

> A sequencia de estados mais provável dado as observações. 
>
> Conhecendo *A*,*B*,*π* é possível calcular a sequência de estados mais provável,  *x*1,...,*xt*,  dado as observações y*1,...,yt. Necessida de criar duas matrices *N*×*T*, onde se guardam as probabilidades dos caminhos mais prováveis de estados para o estado n∊N dadas as primeiras t∊T observações. Podem ser construídas por ordem, a partir do estado anterior
>

##### Baum-Welch algorithm

>Caso particular de E.M usado para descobrir os parametros desconhecidos θ de uma HMM. Tira partido do alogritmo forward-backward para a fase de expectation
>
>- Calcula-se a probabilidade conjunta de estado oculto e emissões até o momento, a cada momento.
>- Calcula-se a probabilidade de emissões futuras, dado o estado oculto a cada momento.
>- Calcula-se o estado oculto de probabilidade a cada momento, dadas todas as emissões.
>- Calcule-se a probabilidade conjunta de estados ocultos consecutivos a cada vez, considerando todas as emissões.
>- Use para encontrar novos valores de parâmetro

### Expectation-Maximization (EM)

> Considere-se um set X de dados conhecidos, um set Z de variáveis desconhecidas, variáveis latentes, e um set de parametros θ que se pretendem ajustar de modo a maximizar a likelihood dos parâmetros. Esta likelihood é a probabilidade de todos os dados, dado o set de parametros θ, L(θ; X, Z) = p(X, Z|θ). Não é possivel calcular a likelihood diretamente por se desconhecer Z, no entanto é possível estimar a distribuição posterior de Z dado X, assumindo suposições à priori de θ. Isto permite calcular um valor esperado para Z e assim estimar os valores necessários para θ que maximizem a likelihood, dado um θ estimado anteriormente.
>
> O algoritmo de K-Means já faz isto. Z corresponde à atribuição dos clusters aos exemplos, X às coordenadas dos exemplos. Dado o set de centroids protótipos anteriores, θ antigo, é possível estimar os melhores valores para Z, atribuindo aos clusters os exemplos mais próximos dos centroids. De seguida já é possível obter a estimativa da likelihood máxima das posições dos novos centroids, o novo θ, ao se calcular o ponto médio de cada cluster. 
>
> Em probabilistic clustering pode-se considerar um modelo probabilistico dos exemplos pertencentes a X, onde se desconhece Y, *P*(*X*,*Y*)=*P*(*X*|*Y*) *P*(*Y*). Atribuindo uma variável z∊Z (variáveis latentes), para cada x ∊ X tal que, cada z tem um valor pertencente a {1,0} e a soma de todos os z é igual a 1, recorrendo à regra de Bayes, é possível calcular as probabilidades posteriores *γ*(znk). Para maximizar a log likelihood ln p(X|π,μ,Σ) pode-se recorrer a E.M, considerando π,μ,Σ como θ. Basta calcular a *γ*(znk) de uma previsão inicial dos valores de θ e usar de novo *γ*(znk) na função de likelohood. maximizando em ordem a θ, repetindo até se obter convergência.



### Contiguity Based Clustering

>A atribuição de uma classe a um dado exemplo depende da geração de redes de exemplos contíguos. Resolve o problema de pre determinar o número de clusters.

#### Affinity Propagation

> Conceptualmente, baseia-se na ideia da propagação de mensagens de "responsabilidade" entre exemplos dos dados, mensagens estas que foram envidadas por cada exemplo aos candidatos a protótipos de clusters. Estas mensagens indicam o quão adequado cada candidato é, de acordo com o exemplo dos dados. São também enviadas mensagens de "disponibilidade", enviadas por cada candidato a todos os pontos dos dados, que indicam o quão adequado o protótipo é com base no suporte que recebeu para ser protótipo.
>
> Inicialmente, quase todos os pontos consideram-se como os melhores protótipos mas, à medida que a disponibilidade e a responsabilidade se propagam, os votos em certos candidatos começam a acumalar. Enventualmente chega-se a uma convergência e ficam gerados os clusters. 

### Density Based Clustering

> A atribuição de uma classe a um dado exemplo depende da densidade dos pontos na região. Regiões com alta densidade de exemplos são considerados clusters, regiões com baixa densidade podem ser consideradas ruído.

#### DBSCAN 

> The DBSCAN (Density-based spatial clustering of applications with noise) baseia-se na definição de vizinhancas de pontos, de tamanho mínimo e a uma distância definidos à priori. Tendo por base os valores de tamanho mínimo e distância, um ponto é um core point se tiver um número de vizinhos superior ao mínimo definido a uma distância inferior à distância definida. Dois pontos são reachable se pertencerem à vizinhança um do outro ou à vizinhança de qualque core point que seja reachable por um dos dois. 
>
> O algoritmo executa do seguinte modo, se um ponto for considerado core point atribui-se um cluster a si e à sua vizinhança, caso contrário é marcado como ruído. Se algum vizinho de um core point é um core point os dois clusters juntam-se.

### Hierarchical Clustering

> A geração de clusters depende da divisão ou junção de clusters já definidos. É possivel definir um hierarquia entre os vários clusters gerados.

#### Divisive Clustering

> Divisive clustering te uma abordagem top-down. Começa com um único cluster com todos os exemplos e, iterativamente, escolhe os melhores dois clusters que dividem os clusters originais. Termina quando o número de clusters é o desejado, ou, no limite, quando só existem singleton clusters, com um único exemplo cada.  O(2 ^n)

##### Bysecting K-Means

> 1. Start with all the examples in a single cluster.
> 2. Choose the best cluster for splitting (e.g. the largest or the one with the lowest score).
> 3. Split the best candidate with k-means, using k = 2.
> 4. Repeat steps 2 and 3 until the desired number of clusters is reached.

#### Agglomerative Clustering

>Agglomerative clustering tem uma abordagem bottom-up. Começa com vários cluster singleton, com um único exemplo, e, iterativamente, vai juntando os dois melhores clusters, dado o linkage method escolhido. Quando todos os clusters estão únidos, termina.  O(n^3)

### Fuzzy Clustering

> Todos os exemplos pertencem a todos os clusters como um valor contínuo de membership, entre 0 e 1.

#### Fuzzy C-Means

> Similar ao K-means mas atribui valores contínuos de membership a cada exemplo. É necessário um método de defuzzification de modo a converter os valores de membership contínuos em valores discretos e exclusivos (1 e 0) para se gerarem os clusters finais.

### Manifold Learning

> Uma n-dimensional manifold, ou n-manifold, consiste num set de pontos tais que cada ponto e os seus vizinhos formam, aproximadamente, um espaço euclideano. Isto é útil em machine learning no sentido de que sets de dado normalmente são conjuntos de pontos que podem ser aproximados por manifolds com n dimensões a menos que as features originais. Descobrir estas n-manifolds permite reduzir a dimensionalidade dos dados.

#### t-SNE

> O método de t-distributed stochastic neighbor embedding algorithm (t-SNE) consiste em projetar dados em dimensões mais reduzidas que as originais, tentanto conservar a distribuição das distâncias entre os pontos de forma análoga. Para tal, dispõe-se os pontos no espaço de features mais reduzido e, com base na distribuição original  das suas distâncias, atraiem-se e repelem-se os pontos de modo a formar grupos. A atração e repulsão dos pontos é em função da distribuição das distâncias originais dos pontos, por exemplo, considerando uma distribuição t-student, de modo a que os pontos conservem as distâncias originais, analogamente, nas novas dimensões mais reduzidas.

#### Isomap

> O algoritmo de Isomap tenta criar manifolds de dimensões mais reduzidas dos dados, tentando conservar a distâncias entre k vizinhos mais próximos. Primeiro, cria um k-nn grafo que conecta cada ponto aos seus k vizinhos mais próximos. Em segundo, calcula as distâncias dos pares de pontos para todos os pontos, tendo por base o caminho mais curto entre os pontos. Finalmente, calcula uma distribuição de pontos no novo espaço de features mais reduzido, tentando conservar ao máximo as distâncias entre todos pares de pontos.
# projetoRP
Repositório do projeto CrimesSF para a disciplina de Reconhecimento de Padrões, do Curso de Bacharelado em Ciência da Computação, da Universidade Federal Rural de Pernambuco, semestre 2015.2.

# Execução do código
Nesta seção serão descritos os passos necessários para conseguir executar o código corretamente.

##Downloads
Antes de mais nada, como requisito para executar o código, você precisa fazer o download das bases de teste e treinamento
disponíveis na [página do Kaggle](https://www.kaggle.com/c/sf-crime/data).

Você vai precisar dos seguintes softwares:

* Linguagem de Programação [Python](https://www.python.org/) instalada no seu computador
* As seguintes bibliotecas para Python:
 * [NumPy](http://www.numpy.org/)
 * [SciPy](http://www.scipy.org/)
 * [Pandas](http://pandas.pydata.org/)
 * [SciKit Learn](http://scikit-learn.org/)
* O [Pentaho Data Integration](http://community.pentaho.com/projects/data-integration/) para execução dos códigos de limpeza e tratamentos dos dados das bases

##Executando

Primeiramente, execute o tratamento de dados.

Navegue até a pasta onde instalou o Pentaho Data Integration e execute o seguinte comando para tratar a base de treinamento:
```
Windows:
> Kitchen.bat /file=<CAMINHO-ATÉ-O-PROJETO>\pdi\job_clean_data.kjb

Linux:
$ kitchen.sh -file=<CAMINHO-ATÉ-O-PROJETO>/pdi/job_clean_data.kjb
```

Depois, faça o mesmo para base de testes.
```
Windows:
> Kitchen.bat /file=<CAMINHO-ATÉ-O-PROJETO>\pdi\job_clean_test_data.kjb

Linux:
$ kitchen.sh -file=<CAMINHO-ATÉ-O-PROJETO>/pdi/job_clean_test_data.kjb
```

Se você quiser verificar os experimentos que estavam sendo realizados antes da publicação do artigo:
```
Windows:
> Kitchen.bat /file=<CAMINHO-ATÉ-O-PROJETO>\pdi\job_clean_data_experiment.kjb

Linux:
$ kitchen.sh -file=<CAMINHO-ATÉ-O-PROJETO>/pdi/job_clean_data_experiment.kjb
```

Por fim, navegue até a pasta do `python`, dentro da pasta raiz do projeto. Ela deve possuir 5 scripts:

* `CSV2ARFF.py` converte o arquivo CSV passado como parâmetro para ARFF. Vale notar que ele não está generalizado 
para qualquer base. Ele acabou ficando obsoleto com a aplicação da técnica de aprendizagem por contagem. O ARFF por ele 
gerado não será válido.
* `CSV2ARFF_Balance.py` tem função similar ao arquivo anterior e também ficou obsoleto. A diferença é que ele 
utilizava o ruído gaussiano para tentar balancear as bases.
* `CSV2ARFFLogODDS.py` também tem como função converter de CSV para ARFF. Este, apesar de não generalizado, deve 
funcionar para a base retornada pelo processo de limpeza e transformação dos dados executado anteriormente.
* `logisticRegression.py` é o script que irá retornar um arquivo CSV que pode ser submetido ao Kaggle como resultado 
da competição.
* `modelTesting.py` é o script de experimento que estava sendo utilizado para experimentar diferentes técnicas e 
algoritmos de aprendizado de máquina.

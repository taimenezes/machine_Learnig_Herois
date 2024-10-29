# Duelo de Heróis: Análise de Pontuação de Filmes Marvel e DC

Este projeto analisa e compara as classificações e bilheteiras dos filmes da Marvel e da DC. Utiliza regressão Lasso para prever a bilheteira mundial com base em características como classificação, metascore, duração e orçamento dos filmes.

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Bootstrap

## Estrutura do Projeto

A estrutura de diretórios do projeto é a seguinte:

/projeto
│
├── /dados
│   └── db.csv                  # Dados sobre os filmes da Marvel e DC
├── /static
│   ├── /css
│   │   └── styles.css          # Estilos personalizados para a interface web
│   └── /images
│       ├── imagem1.png         # Gráfico 1: Classificações de Filmes
│       ├── imagem2.png         # Gráfico 2: 10 filmes de maior bilheteira (Mundo)
│       └── imagem3.png         # Gráfico 3: 10 filmes de maior bilheteira (EUA)
├── app.py                       # Código principal da aplicação Flask
└── requirements.txt             # Lista de dependências do projeto


## Como Executar o Projeto

1. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt

2. **Execute a aplicação flask:**

    ```bash
Copiar código
python app.py

3. **Acesse a aplicação:**

Abra seu navegador e vá para http://127.0.0.1:5000/.

Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para enviar pull requests ou abrir issues.

Licença
Este projeto é licenciado sob a MIT License.


### Instruções

1. **README.md**: Descreve o projeto, as tecnologias usadas, a estrutura do diretório e como executar a aplicação.
2. **.gitignore**: Ignora arquivos e diretórios que não devem ser rastreados pelo Git, como arquivos compilados, ambientes virtuais e arquivos de banco de dados.

Sinta-se à vontade para modificar o conteúdo conforme necessário!
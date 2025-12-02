# Pesquisa: Melhoria do Algoritmo k-means (k-means++)

Este repositório contém o código-fonte e os experimentos realizados para o trabalho final da disciplina de Inteligência Artificial. O projeto consiste em uma análise comparativa rigorosa entre o algoritmo **k-means Padrão** (inicialização aleatória) e o **k-means Melhorado** (inicialização k-means++).

## 📋 Sobre o Projeto

O objetivo deste experimento é demonstrar estatisticamente a superioridade da estratégia de inicialização probabilística (*seeding*) em cenários de alta dimensionalidade.

### Destaques Técnicos
* **Dados Sintéticos Complexos:** Geração de datasets com 10 dimensões (features) para simular problemas reais, em vez de dados 2D simples.
* **Validação Estatística:** O algoritmo é executado 30 vezes para garantir consistência, calculando média e desvio padrão.
* **Métricas Avançadas:**
    * *Inércia (WCSS):* Para medir a compactação dos clusters.
    * *Silhouette Score:* Para medir a qualidade da separação entre os grupos.
    * *Tempo de Convergência:* Eficiência computacional.
* **Visualização Profissional:** Utilização de **PCA (Principal Component Analysis)** para redução de dimensionalidade (10D $\to$ 2D) e plotagem dos resultados.

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Scikit-Learn:** Implementação dos algoritmos de ML e métricas.
* **NumPy & Pandas:** Manipulação algébrica e estruturação dos dados.
* **Matplotlib:** Geração de gráficos científicos.

## 🚀 Como Executar

Siga os passos abaixo para reproduzir os experimentos:

### 1. Instalar Dependências
Certifique-se de ter o Python instalado. Instale as bibliotecas necessárias listadas no `requirements.txt`:

```bash
pip install -r requirements.txt

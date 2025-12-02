import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Configuração visual dos gráficos
plt.style.use('ggplot')

def run_research_experiment():
    print("--- INICIANDO EXPERIMENTO CIENTÍFICO ---")
    
    # Gerar Dataset Complexo (Simulação de dados reais)
    print("[1/4] Gerando dados sintéticos de alta dimensionalidade...")
    n_samples = 3000
    n_features = 10 
    n_clusters = 6
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                      centers=n_clusters, cluster_std=2.5, random_state=42)

    # Configuração da Bateria de Testes
    n_runs = 30 # Rodar 30 vezes para validação estatística
    results = []

    # Executar comparações
    strategies = ['random', 'k-means++']
    
    print(f"[2/4] Executando {n_runs} rodadas para cada algoritmo...")
    
    best_labels = None
    
    for strategy in strategies:
        print(f"   -> Testando estratégia: {strategy.upper()}...")
        
        inertias = []
        times = []
        iterations = []
        
        for i in range(n_runs):
            # NÃO DEIXAR O PADRÃO 10
            kmeans = KMeans(n_clusters=n_clusters, init=strategy, n_init=1, random_state=i)
            
            start_t = time.time()
            kmeans.fit(X)
            end_t = time.time()
            
            inertias.append(kmeans.inertia_)
            times.append(end_t - start_t)
            iterations.append(kmeans.n_iter_)
            
            # Salvar o modelo do K-Means++ para o gráfico final
            if strategy == 'k-means++' and i == 0:
                best_labels = kmeans.labels_

        # Consolidar estatísticas
        avg_inertia = np.mean(inertias)
        std_inertia = np.std(inertias)
        avg_time = np.mean(times)
        avg_iter = np.mean(iterations)
        
        # Calcular Silhouette Score da úlrima rodada
        sil_score = silhouette_score(X, kmeans.labels_)

        results.append({
            'Algoritmo': 'Standard (Random)' if strategy == 'random' else 'Improved (K-Means++)',
            'Inercia_Media': f"{avg_inertia:.2f}",
            'Desvio_Padrao_Inercia': f"{std_inertia:.2f}",
            'Tempo_Medio(s)': f"{avg_time:.4f}",
            'Iteracoes_Media': f"{avg_iter:.1f}",
            'Silhouette_Score': f"{sil_score:.4f}"
        })

    # Salvar Tabela de Resultados
    print("[3/4] Salvando resultados numéricos...")
    df = pd.DataFrame(results)
    df.to_csv("results_summary.csv", index=False)
    print("   -> Tabela salva em 'results_summary.csv'")
    print("\n--- RESULTADOS FINAIS ---")
    print(df)
    print("-------------------------")

    # Gerar Gráfico Profissional com PCA
    print("[4/4] Gerando visualização com PCA (10D -> 2D)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='viridis', s=15, alpha=0.6)
    plt.title('Visualização PCA: Agrupamento com K-Means++ Melhorado')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(label='Cluster')
    
    # Salvar imagem para o artigo
    plt.savefig('cluster_pca_viz.png', dpi=300)
    print("   -> Gráfico salvo em 'cluster_pca_viz.png'")

if __name__ == "__main__":
    run_research_experiment()
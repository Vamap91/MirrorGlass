import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
import pandas as pd
import time
import cv2

# Configuração da página Streamlit
st.set_page_config(
    page_title="Mirror Glass - Detector de Fraudes em Imagens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e introdução
st.title("📊 Mirror Glass: Sistema de Detecção de Fraudes em Imagens")
st.markdown("""
Este sistema utiliza técnicas avançadas de visão computacional para:
1. **Detectar imagens duplicadas** ou altamente semelhantes, mesmo com alterações como cortes ou ajustes
2. **Identificar manipulações por IA** que criam texturas artificialmente uniformes em áreas danificadas

### Como funciona?
1. Faça upload das imagens para análise
2. O sistema analisa duplicidade usando SIFT/SSIM e manipulações de textura usando LBP
3. Resultados são exibidos com detalhamento visual e score de naturalidade
""")

# Classe para análise de texturas
class TextureAnalyzer:
    """
    Classe para análise de texturas usando Local Binary Pattern (LBP).
    Detecta manipulações em imagens automotivas, principalmente restaurações por IA.
    """
    
    def __init__(self, P=8, R=1, block_size=16, threshold=0.3):
        self.P = P  # Número de pontos vizinhos
        self.R = R  # Raio
        self.block_size = block_size  # Tamanho dos blocos para análise
        self.threshold = threshold  # Limiar para textura suspeita
    
    def calculate_lbp(self, image):
        # Converter para escala de cinza e array numpy
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image
            
        # Calcular LBP
        lbp = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        # Calcular histograma de padrões
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Normalização
        
        return lbp, hist
    
    def analyze_texture_variance(self, image):
        # Converter para formato numpy se for PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Cálculo do LBP
        lbp_image, _ = self.calculate_lbp(image)
        
        # Inicializa matriz de variância e entropia
        height, width = lbp_image.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        
        # Analisa blocos da imagem
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = lbp_image[i:i+self.block_size, j:j+self.block_size]
                
                # Calcula a entropia do bloco (medida de aleatoriedade)
                hist, _ = np.histogram(block, bins=10, range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                block_entropy = entropy(hist)
                
                # Normaliza a entropia para um intervalo de 0 a 1
                max_entropy = np.log(10)  # Entropia máxima para 10 bins
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                
                # Calcula variância normalizada
                block_variance = np.var(block) / 255.0
                
                # Armazena nos mapas
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
        
        # Combina entropia e variância para pontuação de naturalidade (70% entropia, 30% variância)
        naturalness_map = entropy_map * 0.7 + variance_map * 0.3
        
        # Normaliza o mapa para visualização
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Cria máscara de áreas suspeitas (baixa naturalidade)
        suspicious_mask = norm_naturalness_map < self.threshold
        
        # Calcula score de naturalidade (0-100)
        naturalness_score = int(np.mean(norm_naturalness_map) * 100)
        
        # Converte para mapa de calor para visualização
        heatmap = cv2.applyColorMap((norm_naturalness_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            "variance_map": variance_map,
            "naturalness_map": norm_naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap
        }
    
    def classify_naturalness(self, score):
        if score <= 30:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 70:
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, analysis_results):
        # Converter para numpy se for PIL
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Extrair resultados
        naturalness_map = analysis_results["naturalness_map"]
        suspicious_mask = analysis_results["suspicious_mask"]
        score = analysis_results["naturalness_score"]
        
        # Redimensionar para o tamanho da imagem original
        height, width = image.shape[:2]
        
        # Redimensionar naturalness_map e suspicious_mask
        naturalness_map_resized = cv2.resize(naturalness_map, 
                                           (width, height), 
                                           interpolation=cv2.INTER_LINEAR)
        
        mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), 
                                 (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Criar mapa de calor
        heatmap = cv2.applyColorMap((naturalness_map_resized * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        
        # Criar overlay com 40% de transparência
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Destacar áreas suspeitas com contorno
        highlighted = overlay.copy()
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)
        
        # Classificar resultado
        category, description = self.classify_naturalness(score)
        
        # Adicionar informações na imagem
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        
        return highlighted, heatmap
    
    def analyze_image(self, image):
        # Analisar textura
        analysis_results = self.analyze_texture_variance(image)
        
        # Gerar visualização
        visual_report, heatmap = self.generate_visual_report(image, analysis_results)
        
        # Classificar o resultado
        score = analysis_results["naturalness_score"]
        category, description = self.classify_naturalness(score)
        
        # Calcular percentual de áreas suspeitas
        percent_suspicious = float(np.mean(analysis_results["suspicious_mask"]) * 100)
        
        # Criar relatório final
        report = {
            "score": score,
            "category": category,
            "description": description,
            "percent_suspicious": percent_suspicious,
            "visual_report": visual_report,
            "heatmap": heatmap,
            "analysis_results": analysis_results
        }
        
        return report

# Barra lateral com controles
st.sidebar.header("⚙️ Configurações")

# Seleção de modo
modo_analise = st.sidebar.radio(
    "Modo de Análise",
    ["Duplicidade", "Manipulação por IA", "Análise Completa"],
    help="Escolha o tipo de análise a ser realizada"
)

# Configurações para detecção de duplicidade
if modo_analise in ["Duplicidade", "Análise Completa"]:
    st.sidebar.subheader("Configurações de Duplicidade")
    limiar_similaridade = st.sidebar.slider(
        "Limiar de Similaridade (%)", 
        min_value=30, 
        max_value=100, 
        value=50, 
        help="Imagens com similaridade acima deste valor serão consideradas possíveis duplicatas"
    )
    limiar_similaridade = limiar_similaridade / 100  # Converter para decimal

    metodo_deteccao = st.sidebar.selectbox(
        "Método de Detecção",
        ["SIFT (melhor para recortes)", "SSIM + SIFT", "SSIM"],
        help="Escolha o método para detectar imagens similares"
    )

# Configurações para detecção de manipulação por IA
if modo_analise in ["Manipulação por IA", "Análise Completa"]:
    st.sidebar.subheader("Configurações de Análise de Textura")
    limiar_naturalidade = st.sidebar.slider(
        "Limiar de Naturalidade", 
        min_value=30, 
        max_value=70, 
        value=50, 
        help="Score abaixo deste valor indica possível manipulação por IA"
    )
    
    tamanho_bloco = st.sidebar.slider(
        "Tamanho do Bloco", 
        min_value=8, 
        max_value=32, 
        value=16, 
        step=4,
        help="Tamanho do bloco para análise de textura (menor = mais sensível)"
    )
    
    threshold_lbp = st.sidebar.slider(
        "Sensibilidade LBP", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.3, 
        step=0.05,
        help="Limiar para detecção de áreas suspeitas (menor = mais sensível)"
    )

# Funções para processamento de imagens - DUPLICIDADE
def preprocessar_imagem(img, tamanho=(300, 300)):
    try:
        # Redimensionar
        img_resize = img.resize(tamanho)
        # Converter para escala de cinza para SSIM
        img_gray = img_resize.convert('L')
        # Converter para array numpy
        img_array = np.array(img_gray)
        # Normalizar valores para [0, 1]
        img_array = img_array / 255.0
        # Converter para CV2 formato (para SIFT)
        img_cv = np.array(img_resize)
        img_cv = img_cv[:, :, ::-1].copy()  # RGB para BGR
        return img_array, img_cv
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None, None

def calcular_similaridade_ssim(img1, img2):
    try:
        # Garantir que as imagens tenham o mesmo tamanho
        if img1.shape != img2.shape:
            img2 = resize(img2, img1.shape)
        
        # Calcular SSIM com data_range especificado
        score = ssim(img1, img2, data_range=1.0)
        return score
    except Exception as e:
        st.error(f"Erro ao calcular similaridade SSIM: {e}")
        return 0

def calcular_similaridade_sift(img1_cv, img2_cv):
    try:
        # Converter para escala de cinza
        img1_gray = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        
        # Inicializar o detector SIFT
        sift = cv2.SIFT_create()
        
        # Detectar keypoints e descritores
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        # Se não houver descritores suficientes, retorna 0
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
            
        # Usar o matcher FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Encontrar os 2 melhores matches para cada descritor
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Filtrar bons matches usando o teste de proporção de Lowe
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Calcular a similaridade baseada no número de bons matches
        max_matches = min(len(kp1), len(kp2))
        if max_matches == 0:
            return 0
            
        similarity = len(good_matches) / max_matches
        
        # Normalizar para evitar valores muito baixos
        if similarity < 0.05:
            adjusted_similarity = 0
        else:
            # Expandir valores pequenos para uma escala mais ampla
            adjusted_similarity = min(1.0, similarity * 2)
        
        return adjusted_similarity
        
    except Exception as e:
        st.error(f"Erro ao calcular similaridade SIFT: {e}")
        return 0

def calcular_similaridade_combinada(img1_gray, img2_gray, img1_cv, img2_cv):
    try:
        # Calcular similaridade usando ambos os métodos
        sim_ssim = calcular_similaridade_ssim(img1_gray, img2_gray)
        sim_sift = calcular_similaridade_sift(img1_cv, img2_cv)
        
        # A similaridade combinada é a média ponderada dos dois valores
        # SIFT tem mais peso para detectar recortes
        return (sim_ssim * 0.3) + (sim_sift * 0.7)
    except Exception as e:
        st.error(f"Erro ao calcular similaridade combinada: {e}")
        return 0

def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_image_download_link(img, filename, text):
    # Converter para PIL Image se for numpy array
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img
        
    # Salvar em buffer
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)
    
    # Codificar para base64
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    
    return href

def visualizar_duplicatas(imagens, nomes, duplicatas, limiar):
    if not duplicatas:
        st.info("Nenhuma duplicata encontrada com o limiar de similaridade atual.")
        return None
    
    # Criar DataFrame para relatório
    relatorio_dados = []
    
    # Para cada grupo de duplicatas
    for idx, (img_orig_idx, similares) in enumerate(duplicatas.items()):
        st.write("---")
        st.subheader(f"Grupo de Duplicatas #{idx+1}")
        
        # Layout para imagem original e suas duplicatas
        cols = st.columns(min(len(similares) + 1, 4))  # Limita a 4 colunas por linha
        
        # Mostrar imagem original
        with cols[0]:
            st.image(imagens[img_orig_idx], caption=f"Original: {nomes[img_orig_idx]}", width=200)
        
        # Mostrar duplicatas
        for i, (similar_idx, similaridade) in enumerate(similares):
            col_index = (i + 1) % len(cols)
            
            # Se precisar de uma nova linha
            if col_index == 0 and i > 0:
                st.write("")  # Linha em branco
                cols = st.columns(min(len(similares) - i + 1, 4))
            
            with cols[col_index]:
                st.image(imagens[similar_idx], width=200)
                caption = f"{nomes[similar_idx]}\nSimilaridade: {similaridade:.2f}"
                st.caption(caption)
                
                # Destacar em verde se acima do limiar
                if similaridade >= limiar:
                    st.success("DUPLICATA DETECTADA")
                
                # Adicionar ao relatório
                relatorio_dados.append({
                    "Arquivo Original": nomes[img_orig_idx],
                    "Arquivo Duplicado": nomes[similar_idx],
                    "Similaridade (%)": round(similaridade * 100, 2)
                })
    
    # Criar DataFrame do relatório
    if relatorio_dados:
        df_relatorio = pd.DataFrame(relatorio_dados)
        return df_relatorio
    return None

# Função principal para detectar duplicatas
def detectar_duplicatas(imagens, nomes, limiar=0.5, metodo="SIFT (melhor para recortes)"):
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processar imagens
    status_text.text("Extraindo características das imagens...")
    arrays_processados_gray = []  # Para SSIM
    arrays_processados_cv = []    # Para SIFT
    indices_validos = []
    
    for i, img in enumerate(imagens):
        # Atualizar barra de progresso
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Processando imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        # Preprocessar imagem
        img_array_gray, img_array_cv = preprocessar_imagem(img)
        if img_array_gray is not None:
            arrays_processados_gray.append(img_array_gray)
            arrays_processados_cv.append(img_array_cv)
            indices_validos.append(i)
    
    if not arrays_processados_gray:
        status_text.error("Nenhuma imagem válida para processamento.")
        return None
    
    # Calcular similaridades
    status_text.text("Comparando imagens e buscando duplicatas...")
    duplicatas = {}  # {índice_original: [(índice_similar, similaridade), ...]}
    
    total_comparacoes = len(arrays_processados_gray) * (len(arrays_processados_gray) - 1) // 2
    comparacao_atual = 0
    
    for i in range(len(arrays_processados_gray)):
        similares = []
        for j in range(len(arrays_processados_gray)):
            # Não comparar uma imagem com ela mesma
            if i != j:
                comparacao_atual += 1
                
                # Atualizar progresso de maneira mais segura
                if total_comparacoes > 0:
                    # Certificar que o progresso sempre está entre 0 e 1
                    progress = min(max(comparacao_atual / total_comparacoes, 0.0), 1.0)
                    progress_bar.progress(progress)
                
                # Calcular similaridade com base no método selecionado
                if metodo == "SSIM":
                    similaridade = calcular_similaridade_ssim(
                        arrays_processados_gray[i], 
                        arrays_processados_gray[j]
                    )
                elif metodo == "SIFT (melhor para recortes)":
                    similaridade = calcular_similaridade_sift(
                        arrays_processados_cv[i], 
                        arrays_processados_cv[j]
                    )
                else:  # SSIM + SIFT
                    similaridade = calcular_similaridade_combinada(
                        arrays_processados_gray[i], 
                        arrays_processados_gray[j],
                        arrays_processados_cv[i], 
                        arrays_processados_cv[j]
                    )
                
                # Se acima do limiar, adicionar como duplicata
                if similaridade >= limiar:
                    similares.append((indices_validos[j], similaridade))
        
        # Se encontrou duplicatas, adicionar à lista
        if similares:
            duplicatas[indices_validos[i]] = similares
    
    progress_bar.empty()
    status_text.text("Processamento concluído!")
    
    return duplicatas

# Funções para análise de manipulação por IA
def analisar_manipulacao_ia(imagens, nomes, limiar_naturalidade=50, tamanho_bloco=16, threshold=0.3):
    # Inicializar analisador de textura
    analyzer = TextureAnalyzer(P=8, R=1, block_size=tamanho_bloco, threshold=threshold)
    
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Resultados
    resultados = []
    
    # Processar cada imagem
    for i, img in enumerate(imagens):
        # Atualizar barra de progresso
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Analisando textura da imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        # Analisar imagem
        report = analyzer.analyze_image(img)
        
        # Adicionar informações ao relatório
        resultados.append({
            "indice": i,
            "nome": nomes[i],
            "score": report["score"],
            "categoria": report["category"],
            "descricao": report["description"],
            "percentual_suspeito": report["percent_suspicious"],
            "visual_report": report["visual_report"],
            "heatmap": report["heatmap"]
        })
    
    progress_bar.empty()
    status_text.text("Análise de textura concluída!")
    
    return resultados

# Função para exibir resultados da análise de textura
def exibir_resultados_textura(resultados):
    if not resultados:
        st.info("Nenhum resultado de análise de textura disponível.")
        return None
    
    # Criar DataFrame para relatório
    relatorio_dados = []
    
    # Para cada imagem analisada
    for res in resultados:
        # Adicionar cabeçalho
        st.write("---")
        st.subheader(f"Análise de Textura: {res['nome']}")
        
        # Layout para exibir resultados
        col1, col2 = st.columns(2)
        
        # Coluna 1: Imagem original e informações
        with col1:
            st.image(res["visual_report"], caption=f"Análise de Textura - {res['nome']}", use_column_width=True)
            
            # Adicionar métricas
            st.metric("Score de Naturalidade", res["score"])
            
            # Status baseado no score
            if res["score"] <= 30:
                st.error(f"⚠️ {res['categoria']}: {res['descricao']}")
            elif res["score"] <= 70:
                st.warning(f"⚠️ {res['categoria']}: {res['descricao']}")
            else:
                st.success(f"✅ {res['categoria']}: {res['descricao']}")
                
            # Download da imagem analisada
            st.markdown(
                get_image_download_link(
                    res["visual_report"], 
                    f"analise_{res['nome'].replace(' ', '_')}.jpg",
                    "📥 Baixar Imagem Analisada"
                ),
                unsafe_allow_html=True
            )
        
        # Coluna 2: Mapa de calor e detalhes
        with col2:
            st.image(res["heatmap"], caption="Mapa de Calor LBP", use_column_width=True)
            
            st.write("### Detalhes da Análise")
            st.write(f"- **Áreas suspeitas:** {res['percentual_suspeito']:.2f}% da imagem")
            st.write(f"- **Interpretação:** {res['descricao']}")
            st.write("- **Legenda do Mapa de Calor:**")
            st.write("  - Azul: Texturas naturais (alta variabilidade)")
            st.write("  - Vermelho: Texturas artificiais (baixa variabilidade)")
            
        # Adicionar ao relatório
        relatorio_dados.append({
            "Arquivo": res["nome"],
            "Score de Naturalidade": res["score"],
            "Categoria": res["categoria"],
            "Percentual Suspeito (%)": round(res["percentual_suspeito"], 2)
        })
    
    # Criar DataFrame do relatório
    if relatorio_dados:
        st.write("---")
        st.write("### Resumo da Análise de Textura")
        df_relatorio = pd.DataFrame(relatorio_dados)
        st.dataframe(df_relatorio)
        
        # Opção para download do relatório
        nome_arquivo = f"relatorio_texturas_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        st.markdown(
            get_csv_download_link(df_relatorio, nome_arquivo, "📥 Baixar Relatório CSV"),
            unsafe_allow_html=True
        )
        
        return df_relatorio
    return None

# Interface principal
st.markdown("### 🔹 Passo 1: Carregar Imagens")
uploaded_files = st.file_uploader(
    "Faça upload das imagens para análise", 
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} imagens carregadas")
    
    # Criar botão para iniciar processamento
    if st.button("🚀 Iniciar Análise", key="iniciar_analise"):
        # Carregar imagens
        imagens = []
        nomes = []
        
for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(img)
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir a imagem {arquivo.name}: {e}")
        
        # Processar de acordo com o modo selecionado
        if modo_analise in ["Duplicidade", "Análise Completa"]:
            try:
                st.markdown("## 🔍 Análise de Duplicidade")
                duplicatas = detectar_duplicatas(imagens, nomes, limiar_similaridade, metodo_deteccao)
                
                # Visualizar resultados de duplicidade
                if duplicatas:
                    # Estatísticas
                    total_duplicatas = sum(len(similares) for similares in duplicatas.values())
                    st.metric("Total de possíveis duplicatas encontradas", total_duplicatas)
                    
                    # Visualizar duplicatas
                    df_relatorio = visualizar_duplicatas(imagens, nomes, duplicatas, limiar_similaridade)
                    
                    # Gerar relatório
                    if df_relatorio is not None:
                        st.markdown("### 🔹 Relatório de Duplicatas")
                        st.dataframe(df_relatorio)
                        
                        # Opção para download do relatório
                        nome_arquivo = f"relatorio_duplicatas_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                        st.markdown(get_csv_download_link(df_relatorio, nome_arquivo, 
                                                     "📥 Baixar Relatório CSV"), unsafe_allow_html=True)
                else:
                    st.warning("Nenhuma duplicata encontrada com o limiar atual. Tente reduzir o limiar de similaridade.")
            except Exception as e:
                st.error(f"Erro durante a detecção de duplicatas: {str(e)}")
        
        # Análise de manipulação por IA
        if modo_analise in ["Manipulação por IA", "Análise Completa"]:
            try:
                st.markdown("## 🤖 Análise de Manipulação por IA")
                resultados_textura = analisar_manipulacao_ia(
                    imagens, 
                    nomes, 
                    limiar_naturalidade,
                    tamanho_bloco,
                    threshold_lbp
                )
                
                # Exibir resultados
                exibir_resultados_textura(resultados_textura)
                
            except Exception as e:
                st.error(f"Erro durante a análise de textura: {str(e)}")
else:
    # Mostrar exemplo quando não há imagens carregadas
    st.info("Faça upload de imagens para começar a detecção de fraudes.")
    
    # Adicionar imagens de exemplo
    if st.button("🔍 Ver exemplos de detecção", key="ver_exemplos"):
        st.write("### Exemplos de Análise de Textura")
        
        # Criar colunas para exibir os exemplos
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://via.placeholder.com/400x300?text=Original", caption="Imagem Original")
            st.write("Score de Naturalidade: 85")
            st.success("✅ Textura Natural")
            
        with col2:
            st.image("https://via.placeholder.com/400x300?text=Manipulada+por+IA", caption="Imagem Manipulada por IA")
            st.write("Score de Naturalidade: 25")
            st.error("⚠️ Alta chance de manipulação")
            
        st.write("### Exemplo de Detecção de Duplicidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://via.placeholder.com/400x300?text=Original", caption="Imagem Original")
            
        with col2:
            st.image("https://via.placeholder.com/400x300?text=Duplicata+Recortada", caption="Duplicata (Recortada)")
            st.write("Similaridade: 0.78")
            st.success("DUPLICATA DETECTADA")

# Rodapé
st.markdown("---")
st.markdown("### Como interpretar os resultados")

# Explicação sobre duplicidade
if modo_analise in ["Duplicidade", "Análise Completa"]:
    st.write("""
    **Análise de Duplicidade:**
    - **Similaridade 100%**: Imagens idênticas
    - **Similaridade >90%**: Praticamente idênticas (possivelmente recortadas ou com filtros)
    - **Similaridade 70-90%**: Muito semelhantes (potenciais duplicatas)
    - **Similaridade 50-70%**: Semelhantes (verificar manualmente)
    - **Similaridade 30-50%**: Possivelmente relacionadas (verificar com atenção)
    - **Similaridade <30%**: Provavelmente não são duplicatas
    """)

# Explicação sobre análise de textura
if modo_analise in ["Manipulação por IA", "Análise Completa"]:
    st.write("""
    **Análise de Manipulação por IA:**
    - **Score 0-30**: Alta probabilidade de manipulação por IA
    - **Score 31-70**: Textura suspeita, requer verificação manual
    - **Score 71-100**: Textura natural, baixa probabilidade de manipulação
    
    O mapa de calor mostra áreas com baixa variância de textura (vermelho) que são típicas 
    de restaurações por IA, onde a textura é artificialmente uniforme.
    """)

# Contato e informações
st.sidebar.markdown("---")
st.sidebar.info("""
**Desenvolvido para:** Mirror Glass
**Projeto:** Detecção de Fraudes em Imagens Automotivas
**Versão:** 1.1.0 (Maio/2025)
""")

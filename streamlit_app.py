import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage import img_as_float
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
import pandas as pd
import time

# Configuração da página Streamlit
st.set_page_config(
    page_title="Detector de Imagens Duplicadas",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e introdução
st.title("📊 Sistema de Detecção de Fraudes em Imagens")
st.markdown("""
Este sistema utiliza técnicas de visão computacional para detectar imagens duplicadas ou altamente semelhantes, 
mesmo com pequenas alterações como cortes, ajustes de brilho ou espelhamento.

### Como funciona?
1. Faça upload das imagens para análise
2. O sistema extrai características das imagens usando múltiplos métodos
3. Automaticamente compara as imagens entre si
4. Identifica possíveis duplicatas baseadas no limiar de similaridade definido
""")

# Barra lateral com controles
st.sidebar.header("⚙️ Configurações")
limiar_similaridade = st.sidebar.slider(
    "Limiar de Similaridade (%)", 
    min_value=50, 
    max_value=100, 
    value=70, 
    help="Imagens com similaridade acima deste valor serão consideradas possíveis duplicatas"
)
limiar_similaridade = limiar_similaridade / 100  # Converter para decimal

# Método de detecção
metodo_deteccao = st.sidebar.selectbox(
    "Método de Detecção",
    ["Combinado (recomendado)", "SSIM", "Detector de características (ORB)"],
    help="Escolha o método para detectar imagens similares"
)

# Funções para processamento de imagens
def preprocessar_imagem(img, tamanho=(224, 224)):
    """Pré-processa uma imagem para análise"""
    try:
        # Redimensionar
        img_resize = img.resize(tamanho)
        # Converter para escala de cinza para simplificar a comparação
        img_gray = img_resize.convert('L')
        # Converter para array numpy
        img_array = np.array(img_gray)
        # Normalizar valores para [0, 1]
        img_array = img_array / 255.0
        return img_array, np.array(img_resize)
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None, None

def calcular_similaridade_ssim(img1, img2):
    """Calcula a similaridade entre duas imagens usando SSIM"""
    try:
        # Garantir que as imagens tenham o mesmo tamanho
        if img1.shape != img2.shape:
            img2 = resize(img2, img1.shape)
        
        # Calcular SSIM com data_range especificado
        # As imagens são normalizadas para [0, 1], então o data_range é 1.0
        score = ssim(img1, img2, data_range=1.0)
        return score
    except Exception as e:
        st.error(f"Erro ao calcular similaridade SSIM: {e}")
        return 0

def calcular_similaridade_orb(img1_color, img2_color):
    """
    Calcula a similaridade entre duas imagens usando o detector de características ORB
    Este método é mais robusto para detectar imagens recortadas
    """
    try:
        # Converter para escala de cinza
        img1 = rgb2gray(img1_color)
        img2 = rgb2gray(img2_color)
        
        # Inicializar o detector ORB
        orb = ORB(n_keypoints=100)
        
        # Extrair características da primeira imagem
        orb.detect_and_extract(img1)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors
        
        # Extrair características da segunda imagem
        orb.detect_and_extract(img2)
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors
        
        # Se não for possível extrair características, retorna 0
        if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
            return 0
        
        # Encontrar correspondências entre as características
        matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
        
        # Calcular a similaridade baseada no número de correspondências
        # em relação ao número total de pontos-chave
        max_keypoints = max(len(keypoints1), len(keypoints2))
        if max_keypoints == 0:
            return 0
            
        # A similaridade é a proporção de correspondências
        similarity = len(matches) / max_keypoints
        
        return similarity
    except Exception as e:
        st.error(f"Erro ao calcular similaridade ORB: {e}")
        return 0

def calcular_similaridade_combinada(img1_gray, img2_gray, img1_color, img2_color):
    """
    Combina os métodos SSIM e ORB para obter uma detecção mais robusta
    """
    try:
        # Calcular similaridade usando ambos os métodos
        sim_ssim = calcular_similaridade_ssim(img1_gray, img2_gray)
        sim_orb = calcular_similaridade_orb(img1_color, img2_color)
        
        # A similaridade combinada é o máximo dos dois valores
        # Isso permite detectar duplicatas mesmo que apenas um dos métodos tenha sucesso
        return max(sim_ssim, sim_orb)
    except Exception as e:
        st.error(f"Erro ao calcular similaridade combinada: {e}")
        return 0

def get_csv_download_link(df, filename, text):
    """Gera link para download de CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def visualizar_duplicatas(imagens, nomes, duplicatas, limiar):
    """Visualiza as duplicatas encontradas em formato de grid"""
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
def detectar_duplicatas(imagens, nomes, limiar=0.7, metodo="Combinado (recomendado)"):
    """Detecta duplicatas entre as imagens carregadas"""
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processar imagens
    status_text.text("Extraindo características das imagens...")
    arrays_processados_gray = []  # Para SSIM
    arrays_processados_color = []  # Para ORB
    indices_validos = []
    
    for i, img in enumerate(imagens):
        # Atualizar barra de progresso
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Processando imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        # Preprocessar imagem
        img_array_gray, img_array_color = preprocessar_imagem(img)
        if img_array_gray is not None:
            arrays_processados_gray.append(img_array_gray)
            arrays_processados_color.append(img_array_color)
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
                elif metodo == "Detector de características (ORB)":
                    similaridade = calcular_similaridade_orb(
                        arrays_processados_color[i], 
                        arrays_processados_color[j]
                    )
                else:  # Combinado (padrão)
                    similaridade = calcular_similaridade_combinada(
                        arrays_processados_gray[i], 
                        arrays_processados_gray[j],
                        arrays_processados_color[i], 
                        arrays_processados_color[j]
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
    if st.button("🚀 Iniciar Detecção de Duplicatas"):
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
        
        # Detectar duplicatas
        try:
            duplicatas = detectar_duplicatas(imagens, nomes, limiar_similaridade, metodo_deteccao)
            
            # Visualizar resultados
            if duplicatas:
                st.markdown("### 🔹 Resultados da Detecção")
                
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
else:
    # Mostrar exemplo quando não há imagens carregadas
    st.info("Faça upload de imagens para começar a detecção de duplicatas.")
    
    # Adicionar imagens de exemplo
    if st.button("🔍 Ver imagens de exemplo"):
        # Criar colunas para exibir as imagens de exemplo
        cols = st.columns(3)
        
        # Criar imagens de exemplo
        with cols[0]:
            st.image("https://via.placeholder.com/300x200?text=Exemplo+1", caption="Exemplo 1")
        with cols[1]:
            st.image("https://via.placeholder.com/300x200?text=Exemplo+2", caption="Exemplo 2")
        with cols[2]:
            st.image("https://via.placeholder.com/300x200?text=Exemplo+3", caption="Exemplo 3")
        
        st.write("Nota: As imagens acima são apenas exemplos visuais. Faça upload de suas próprias imagens para análise.")

# Rodapé
st.markdown("---")
st.markdown("### Como interpretar os resultados")
st.write("""
- **Similaridade 100%**: Imagens idênticas
- **Similaridade >90%**: Praticamente idênticas (possivelmente recortadas ou com filtros)
- **Similaridade 70-90%**: Muito semelhantes (potenciais duplicatas)
- **Similaridade 50-70%**: Semelhantes (verificar manualmente)
- **Similaridade <50%**: Provavelmente não são duplicatas
""")

# Contato e informações
st.sidebar.markdown("---")
st.sidebar.info("""
**Desenvolvido para:** Carglass Automotiva
**Projeto:** Detecção de Fraudes em Pedidos com Imagens Duplicadas
""")

# app.py - Aplicação Streamlit para Detecção de Fraudes em Imagens
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import faiss
from tqdm import tqdm
import pickle
import time
import pandas as pd
import io
import base64

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
Este sistema utiliza Inteligência Artificial para detectar imagens duplicadas ou altamente semelhantes, 
mesmo com pequenas alterações como cortes, ajustes de brilho ou espelhamento.

### Como funciona?
1. Faça upload das imagens para análise
2. O sistema extrai "assinaturas digitais" (embeddings) das imagens
3. Automaticamente compara as imagens entre si
4. Identifica possíveis duplicatas baseadas no limiar de similaridade definido
""")

# Barra lateral com controles
st.sidebar.header("⚙️ Configurações")
limiar_similaridade = st.sidebar.slider(
    "Limiar de Similaridade (%)", 
    min_value=70, 
    max_value=100, 
    value=90, 
    help="Imagens com similaridade acima deste valor serão consideradas possíveis duplicatas"
)
limiar_similaridade = limiar_similaridade / 100  # Converter para decimal

# Funções do sistema
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo ResNet50 para extração de características"""
    with st.spinner("Carregando modelo de IA (isso pode levar alguns segundos)..."):
        modelo_base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return modelo_base

def preprocessar_imagem(img, tamanho=(224, 224)):
    """Pré-processa uma imagem para o modelo"""
    try:
        img = img.resize(tamanho)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None

def extrair_embedding(modelo, imagem_processada):
    """Extrai o embedding de uma imagem"""
    if imagem_processada is not None:
        embedding = modelo.predict(imagem_processada, verbose=0)
        # Normalizar (obrigatório para FAISS IndexFlatIP)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()
    return None

def criar_indice_faiss(embeddings):
    """Cria um índice FAISS para busca rápida"""
    dimensao = embeddings.shape[1]
    indice = faiss.IndexFlatIP(dimensao)  # Produto interno (similaridade de cosseno para vetores normalizados)
    indice.add(embeddings)
    return indice

def buscar_similares(indice, embedding_consulta, k=5):
    """Busca as k imagens mais similares"""
    if embedding_consulta.ndim == 1:
        embedding_consulta = np.expand_dims(embedding_consulta, axis=0)
    distancias, indices = indice.search(embedding_consulta, k)
    return distancias, indices

def get_image_download_link(img, filename, text):
    """Gera link para download de imagem"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

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
        cols = st.columns(len(similares) + 1)
        
        # Mostrar imagem original
        with cols[0]:
            st.image(imagens[img_orig_idx], caption=f"Original: {nomes[img_orig_idx]}", width=200)
        
        # Mostrar duplicatas
        for i, (similar_idx, similaridade) in enumerate(similares):
            with cols[i+1]:
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
def detectar_duplicatas(imagens, nomes, limiar=0.9):
    """Detecta duplicatas entre as imagens carregadas"""
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Carregar modelo
    modelo = carregar_modelo()
    
    # Processar imagens
    status_text.text("Extraindo características das imagens...")
    embeddings = []
    processados = []
    
    for i, img in enumerate(imagens):
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Processando imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        img_processada = preprocessar_imagem(img)
        if img_processada is not None:
            embedding = extrair_embedding(modelo, img_processada)
            if embedding is not None:
                embeddings.append(embedding)
                processados.append(i)
    
    if not embeddings:
        status_text.error("Nenhuma imagem válida para processamento.")
        return None
    
    # Converter para array numpy
    embeddings = np.array(embeddings)
    
    # Criar índice FAISS
    status_text.text("Criando índice para busca rápida...")
    indice = criar_indice_faiss(embeddings)
    
    # Buscar duplicatas
    status_text.text("Comparando imagens e buscando duplicatas...")
    duplicatas = {}  # {índice_original: [(índice_similar, similaridade), ...]}
    
    for i, idx in enumerate(processados):
        progress = (i + 1) / len(processados)
        progress_bar.progress(progress)
        
        # Buscar similares (exceto a própria imagem)
        distancias, indices = buscar_similares(indice, embeddings[i], k=len(embeddings))
        
        # Filtrar similares acima do limiar (excluindo a própria imagem)
        similares = []
        for j in range(1, len(indices[0])):  # Começa de 1 para ignorar a própria imagem
            if distancias[0][j] >= limiar:
                similar_global_idx = processados[indices[0][j]]
                similares.append((similar_global_idx, distancias[0][j]))
        
        if similares:
            duplicatas[idx] = similares
    
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
        duplicatas = detectar_duplicatas(imagens, nomes, limiar_similaridade)
        
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
            st.info("Nenhuma duplicata encontrada com o limiar atual. Tente reduzir o limiar de similaridade.")
else:
    # Mostrar exemplo quando não há imagens carregadas
    st.info("Faça upload de imagens para começar a detecção de duplicatas.")
    
    # Opção de demonstração com imagens de exemplo
    if st.button("🔍 Ver demonstração com imagens de exemplo"):
        st.write("Função de demonstração ainda não implementada. Por favor, faça upload de suas próprias imagens.")

# Rodapé
st.markdown("---")
st.markdown("### Como interpretar os resultados")
st.write("""
- **Similaridade 100%**: Imagens idênticas
- **Similaridade >95%**: Praticamente idênticas (possivelmente recortadas ou com filtros)
- **Similaridade 90-95%**: Muito semelhantes (potenciais duplicatas)
- **Similaridade 80-90%**: Semelhantes (verificar manualmente)
- **Similaridade <80%**: Provavelmente não são duplicatas
""")

# Contato e informações
st.sidebar.markdown("---")
st.sidebar.info("""
**Desenvolvido para:** Carglass Automotiva
**Projeto:** Detecção de Fraudes em Pedidos com Imagens Duplicadas
""")

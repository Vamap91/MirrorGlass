import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import json
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy, kurtosis
import pandas as pd
import time
import cv2
from sklearn.cluster import KMeans

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

# Classe para análise de texturas melhorada
class TextureAnalyzer:
    """
    Classe para análise de texturas usando Local Binary Pattern (LBP).
    Detecta manipulações em imagens automotivas, principalmente restaurações por IA.
    """
    
    def __init__(self, P=8, R=1, block_size=8, threshold=0.10):
        self.P = P  # Número de pontos vizinhos
        self.R = R  # Raio
        self.block_size = block_size  # Tamanho dos blocos para análise
        self.threshold = threshold  # Limiar para textura suspeita
        self.scales = [0.5, 1.0, 2.0]  # Múltiplas escalas para análise
    
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
        
        return lbp, hist, img_gray
    
    def analyze_texture_variance(self, image):
        """
        Versão especializada para detecção de manipulações por IA em imagens de veículos
        """
        # Converter para formato numpy se for PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Converter para escala de cinza
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        
        # 1. Detecção de bordas usando Sobel e Canny
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude do gradiente
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Detecção de bordas com Canny
        edges = cv2.Canny(img_gray, 50, 150)
        
        # 2. Aplicar filtro de mediana para reduzir ruído
        img_filtered = cv2.medianBlur(img_gray, 5)
        
        # 3. Calcular LBP em múltiplas escalas
        lbp_maps = []
        blurred_maps = []
        
        for scale in self.scales:
            # Redimensionar para a escala atual
            if scale != 1.0:
                height, width = img_gray.shape
                new_height, new_width = int(height * scale), int(width * scale)
                img_scaled = cv2.resize(img_gray, (new_width, new_height))
                # Aplicar blurring para simular diferentes níveis de detalhes
                blurred = cv2.GaussianBlur(img_scaled, (5, 5), 0)
                lbp_scaled, _, _ = self.calculate_lbp(blurred)
                # Redimensionar de volta para tamanho original
                lbp_map = cv2.resize(lbp_scaled, (width, height))
            else:
                lbp_map, _, _ = self.calculate_lbp(img_gray)
            
            lbp_maps.append(lbp_map)
            
            # Filtro Gaussiano em diferentes escalas para detectar áreas suspeitas
            for sigma in [1, 3, 5]:
                blurred = cv2.GaussianBlur(img_gray, (sigma*2+1, sigma*2+1), sigma)
                blurred_maps.append(blurred)
        
        # 4. Análise de textura em blocos
        height, width = img_gray.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        gradient_consistency_map = np.zeros((rows, cols))
        edge_density_map = np.zeros((rows, cols))
        blur_consistency_map = np.zeros((rows, cols))
        
        # 5. Analisar em blocos
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                # Extrair blocos
                block_gray = img_gray[i:i+self.block_size, j:j+self.block_size]
                if i < lbp_maps[1].shape[0] - self.block_size and j < lbp_maps[1].shape[1] - self.block_size:
                    block_lbp = lbp_maps[1][i:i+self.block_size, j:j+self.block_size]  # Escala padrão
                else:
                    continue  # Pular blocos que estão fora dos limites
                
                block_gradient = gradient_magnitude[i:i+self.block_size, j:j+self.block_size]
                block_edges = edges[i:i+self.block_size, j:j+self.block_size]
                
                # Calcular entropia LBP (mede aleatoriedade da textura)
                hist, _ = np.histogram(block_lbp, bins=10, range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                block_entropy = entropy(hist)
                max_entropy = np.log(10)
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                
                # Variância da textura (texturas naturais são mais variadas)
                block_variance = np.var(block_lbp) / 255.0
                
                # Consistência do gradiente (gradientes naturais são menos regulares)
                grad_hist, _ = np.histogram(block_gradient, bins=8)
                grad_hist = grad_hist.astype("float")
                grad_hist /= (grad_hist.sum() + 1e-7)
                grad_entropy = entropy(grad_hist)
                grad_consistency = 1.0 - (grad_entropy / np.log(8))  # Normalizado e invertido
                
                # Densidade de bordas (áreas restauradas têm menos bordas naturais)
                edge_density = np.sum(block_edges > 0) / (self.block_size * self.block_size)
                
                # Consistência do blur (resposta a diferentes níveis de borramento)
                blur_responses = []
                for blurred in blurred_maps:
                    blur_block = blurred[i:i+self.block_size, j:j+self.block_size]
                    # Diferença entre original e borrado
                    diff = np.abs(block_gray.astype(float) - blur_block.astype(float)).mean()
                    blur_responses.append(diff)
                
                # O desvio padrão das respostas mede a naturalidade
                # Texturas reais têm resposta mais variada ao blurring
                blur_consistency = 1.0 - min(np.std(blur_responses) / 10.0, 1.0)  # Normalizado e invertido
                
                # Armazenar nos mapas
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
                    gradient_consistency_map[row_idx, col_idx] = grad_consistency
                    edge_density_map[row_idx, col_idx] = edge_density
                    blur_consistency_map[row_idx, col_idx] = blur_consistency
        
        # 6. Análise específica para carros (grandes áreas planas com textura uniforme)
        # Converter para espaço de cor LAB para análise mais perceptual
        if len(image.shape) > 2:
            try:
                lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_channel = lab_image[:,:,0]  # Canal de luminância
                
                # Detectar áreas de luminância semelhante
                luminance_variance = np.zeros((rows, cols))
                for i in range(0, height - self.block_size + 1, self.block_size):
                    for j in range(0, width - self.block_size + 1, self.block_size):
                        if i < l_channel.shape[0] - self.block_size and j < l_channel.shape[1] - self.block_size:
                            block_l = l_channel[i:i+self.block_size, j:j+self.block_size]
                            row_idx = i // self.block_size
                            col_idx = j // self.block_size
                            if row_idx < rows and col_idx < cols:
                                luminance_variance[row_idx, col_idx] = np.var(block_l)
                
                # Normalizar variância de luminância
                if np.max(luminance_variance) > 0:
                    luminance_variance = cv2.normalize(luminance_variance, None, 0, 1, cv2.NORM_MINMAX)
                
                # Áreas com baixa variância de luminância e baixa entropia de textura
                # são candidatas fortes para manipulação por IA
                flat_surface_map = (1.0 - luminance_variance) * (1.0 - entropy_map)
            except Exception as e:
                # Em caso de erro, criar um mapa vazio
                flat_surface_map = np.zeros_like(entropy_map)
        else:
            flat_surface_map = np.zeros_like(entropy_map)
        
        # 7. Detecção de padrões repetitivos (característico de IA)
        # Implementação corrigida que evita o erro de matchTemplate
        lbp_main = lbp_maps[1]  # Escala padrão
        repetitive_pattern_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                if i >= lbp_main.shape[0] - self.block_size or j >= lbp_main.shape[1] - self.block_size:
                    continue  # Pular se fora dos limites
                    
                block = lbp_main[i:i+self.block_size, j:j+self.block_size].copy()
                
                # Verificar se o bloco tem valores válidos
                if np.isfinite(block).all() and np.any(block != 0):
                    try:
                        # Garantir que seja float32 para matchTemplate
                        block_float = block.astype(np.float32)
                        
                        # Calcular a autocorrelação de maneira simplificada e robusta
                        # Criar versão suavizada para análise de textura
                        block_smooth = cv2.GaussianBlur(block_float, (3, 3), 0)
                        
                        # Calcular a variação da textura de forma mais robusta
                        texel_variation = np.std(block_smooth) / np.mean(block_smooth) if np.mean(block_smooth) > 0 else 0
                        
                        # Texturas artificiais têm variação mais baixa (fator invertido)
                        repetitive_score = max(0, 1.0 - min(texel_variation * 2, 1.0))
                    except Exception as e:
                        # Em caso de erro, atribuir valor médio neutro
                        repetitive_score = 0.5
                else:
                    repetitive_score = 0.5
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                if row_idx < rows and col_idx < cols:
                    repetitive_pattern_map[row_idx, col_idx] = repetitive_score
        
        # 8. Combinar todas as métricas para score final
        # Pesos de cada métrica (ajustados especificamente para carros)
        weights = {
            'entropy': 0.15,            # Aleatoriedade da textura
            'variance': 0.10,           # Variação da textura
            'gradient': 0.10,           # Regularidade do gradiente
            'edge_density': 0.15,       # Densidade de bordas naturais
            'blur_consistency': 0.20,   # Resposta a diferentes níveis de blur
            'flat_surface': 0.20,       # Áreas planas com textura uniforme
            'repetitive': 0.10          # Padrões repetitivos
        }
        
        # Invertemos algumas métricas para que valores maiores indiquem manipulação
        naturalness_map = (
            (1.0 - weights['entropy'] * (1.0 - entropy_map)) *          # Entropia (maior é melhor)
            (1.0 - weights['variance'] * (1.0 - variance_map)) *        # Variância (maior é melhor)
            (1.0 - weights['gradient'] * gradient_consistency_map) *    # Consistência do gradiente (menor é melhor)
            (1.0 - weights['edge_density'] * (1.0 - edge_density_map)) * # Densidade de bordas (maior é melhor)
            (1.0 - weights['blur_consistency'] * blur_consistency_map) * # Consistência do blur (menor é melhor)
            (1.0 - weights['flat_surface'] * flat_surface_map) *        # Superfícies planas artificiais (menor é melhor)
            (1.0 - weights['repetitive'] * repetitive_pattern_map)      # Padrões repetitivos (menor é melhor)
        )
        
        # Normalizar mapa para visualização
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # 9. Aplicar threshold mais baixo em áreas de veículos (carroceria)
        # As áreas planas precisam de mais sensibilidade
        vehicle_regions = flat_surface_map > 0.5  # Áreas prováveis de carroceria
        adjusted_threshold_map = np.ones_like(norm_naturalness_map) * self.threshold
        adjusted_threshold_map[vehicle_regions] = self.threshold * 0.7  # 30% mais sensível
        
        # Cria máscara de áreas suspeitas usando threshold adaptativo
        suspicious_mask = norm_naturalness_map < adjusted_threshold_map
        
        # 10. Calcular score de naturalidade (0-100)
        naturalness_score = int(np.mean(norm_naturalness_map) * 100)
        
        # Penalizar scores para imagens de veículos com grandes áreas suspeitas
        if np.sum(vehicle_regions) > 0.2 * rows * cols:  # Se mais de 20% da imagem for veículo
            if np.mean(suspicious_mask[vehicle_regions]) > 0.3:  # Se mais de 30% das áreas de veículo forem suspeitas
                naturalness_score = max(10, naturalness_score - 30)  # Reduzir score em 30 pontos (mínimo 10)
        
        # 11. Converte para mapa de calor para visualização
        heatmap = cv2.applyColorMap(
            (norm_naturalness_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        return {
            "naturalness_map": norm_naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "entropy_map": entropy_map,
            "variance_map": variance_map,
            "gradient_map": gradient_consistency_map,
            "edge_map": edge_density_map,
            "blur_map": blur_consistency_map,
            "flat_surface_map": flat_surface_map,
            "repetitive_map": repetitive_pattern_map,
            "weights": weights,
            "vehicle_regions": vehicle_regions,
            "percentage_suspicious": float(np.mean(suspicious_mask) * 100)
        }
    
    def classify_naturalness(self, score):
        """
        Classificação ajustada para maior sensibilidade
        """
        if score <= 45:  # Limiar mais alto para manipulação
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 70:  # Faixa mais ampla para suspeita
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def analyze_image(self, image):
        # Inicializa um relatório padrão com valores seguros
        report = {
            "score": 0,
            "category": "Erro",
            "description": "Falha na análise inicial",
            "percentual_suspeito": 0,
            "analysis_results": {}
        }
        
        try:
            # Analisar textura
            analysis_results = self.analyze_texture_variance(image)
            if analysis_results is None:
                raise ValueError("analyze_texture_variance retornou None")
            report["analysis_results"] = analysis_results

            # Classificar o resultado
            score = analysis_results.get("naturalness_score", 0)
            report["score"] = score
            category, description = self.classify_naturalness(score)
            report["category"] = category
            report["description"] = description

            # Calcular percentual de áreas suspeitas
            suspicious_mask = analysis_results.get("suspicious_mask")
            if suspicious_mask is not None:
                report["percentual_suspeito"] = float(np.mean(suspicious_mask) * 100)
            else:
                report["percentual_suspeito"] = 0.0

            return report
        except Exception as e:
            # Atualiza a descrição do erro no report padrão
            report["description"] = f"Erro na análise de imagem: {str(e)}"
            # Retorna o dicionário de erro padronizado
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

# Configurações para detecção de manipulação por IA
if modo_analise in ["Manipulação por IA", "Análise Completa"]:
   st.sidebar.subheader("Configurações de Análise de Textura")
   limiar_naturalidade = st.sidebar.slider(
       "Limiar de Naturalidade", 
       min_value=30, 
       max_value=80, 
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
       value=0.35, 
       step=0.05,
       help="Limiar para detecção de áreas suspeitas (menor = mais sensível)"
   )

# Opção para formato de saída
st.sidebar.subheader("Formato de Saída")
formato_saida = st.sidebar.radio(
    "Escolha o formato de resultado",
    ["Interface Visual", "JSON Puro"],
    help="Interface Visual mostra os resultados na tela, JSON Puro retorna apenas os dados da análise"
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

def detectar_duplicatas(imagens, nomes, limiar=0.5):
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
       progress_bar.empty()
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
               
               # Calcular similaridade com método combinado SSIM + SIFT
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
def analisar_manipulacao_ia(imagens, nomes, limiar_naturalidade=50, tamanho_bloco=16, threshold=0.35):
    # Inicializar analisador de textura com parâmetros atualizados
    analyzer = TextureAnalyzer(P=8, R=1, block_size=tamanho_bloco, threshold=threshold)
    
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Resultados
    resultados = []
    
    # Processar cada imagem individualmente
    for i, img in enumerate(imagens):
        # Atualizar barra de progresso
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Analisando textura da imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        try:
            # Analisar imagem individualmente
            report = analyzer.analyze_image(img)
            
            # Validação adicional para garantir que report não é None
            if report is None:
                st.error(f"Erro crítico: analyze_image retornou None para {nomes[i]}")
                resultados.append({
                    "indice": i, 
                    "nome": nomes[i], 
                    "score": 0,
                    "categoria": "Erro Crítico", 
                    "descricao": "Falha interna na análise",
                    "percentual_suspeito": 0,
                    "analysis_results": {}
                })
                continue  # Pula para a próxima imagem
            
            # Adicionar informações ao relatório (agora com acesso mais seguro)
            resultados.append({
                "indice": i,
                "nome": nomes[i],
                "score": report.get("score", 0),
                "categoria": report.get("category", "Erro"),
                "descricao": report.get("description", "N/A"),
                "percentual_suspeito": report.get("percentual_suspeito", 0),
                "analysis_results": report.get("analysis_results", {})
            })
        except Exception as e:
            st.error(f"Erro ao analisar imagem {nomes[i]}: {str(e)}")
            # Adicionar um relatório vazio para manter a consistência
            resultados.append({
                "indice": i,
                "nome": nomes[i],
                "score": 0,
                "categoria": "Erro na análise",
                "descricao": f"Erro: {str(e)}",
                "percentual_suspeito": 0,
                "analysis_results": {}
            })
    
    progress_bar.empty()
    status_text.text("Análise de textura concluída!")
    
    return resultados

# Função para converter numpy arrays para listas (para JSON)
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj

# Função para exibir JSON puro
def exibir_json_puro(dados, tipo_analise):
    st.subheader(f"📄 Resultado JSON - {tipo_analise}")
    
    # Converter numpy arrays para listas
    dados_json = convert_numpy_to_list(dados)
    
    # Mostrar JSON formatado
    json_str = json.dumps(dados_json, indent=2, ensure_ascii=False)
    st.code(json_str, language='json')
    
    # Botão para download do JSON
    st.download_button(
        label="📥 Baixar JSON",
        data=json_str,
        file_name=f"analise_{tipo_analise.lower().replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Função para exibir interface visual (simplificada)
def exibir_interface_visual(dados, tipo_analise):
    if tipo_analise == "Duplicidade":
        if dados:
            # Estatísticas
            total_duplicatas = sum(len(similares) for similares in dados.values())
            st.metric("Total de possíveis duplicatas encontradas", total_duplicatas)
            st.success("✅ Duplicatas detectadas! Verifique os detalhes no JSON.")
        else:
            st.warning("Nenhuma duplicata encontrada com o limiar atual.")
            
    elif tipo_analise == "Manipulação por IA":
        if dados:
            # Resumo dos resultados
            total_imagens = len(dados)
            manipuladas = sum(1 for item in dados if item["score"] <= 45)
            suspeitas = sum(1 for item in dados if 45 < item["score"] <= 70)
            naturais = sum(1 for item in dados if item["score"] > 70)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Imagens", total_imagens)
            with col2:
                st.metric("Manipuladas", manipuladas)
            with col3:
                st.metric("Suspeitas", suspeitas)
            with col4:
                st.metric("Naturais", naturais)
            
            # Lista resumida dos resultados
            st.subheader("Resumo da Análise")
            for item in dados:
                score = item["score"]
                nome = item["nome"]
                
                if score <= 45:
                    st.error(f"⚠️ {nome}: Score {score} - {item['categoria']}")
                elif score <= 70:
                    st.warning(f"⚠️ {nome}: Score {score} - {item['categoria']}")
                else:
                    st.success(f"✅ {nome}: Score {score} - {item['categoria']}")

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
        
        resultados_finais = {}
        
        # Processar de acordo com o modo selecionado
        if modo_analise in ["Duplicidade", "Análise Completa"]:
            try:
                st.markdown("## 🔍 Análise de Duplicidade")
                duplicatas = detectar_duplicatas(imagens, nomes, limiar_similaridade)
                
                # Preparar dados estruturados para JSON
                duplicatas_estruturadas = []
                if duplicatas:
                    for img_orig_idx, similares in duplicatas.items():
                        grupo = {
                            "imagem_original": {
                                "indice": img_orig_idx,
                                "nome": nomes[img_orig_idx]
                            },
                            "duplicatas_encontradas": [
                                {
                                    "indice": similar_idx,
                                    "nome": nomes[similar_idx],
                                    "similaridade": float(similaridade),
                                    "similaridade_percentual": round(similaridade * 100, 2)
                                }
                                for similar_idx, similaridade in similares
                            ]
                        }
                        duplicatas_estruturadas.append(grupo)
                
                resultados_finais["duplicidade"] = {
                    "metodo_usado": "SSIM + SIFT",
                    "limiar_similaridade": limiar_similaridade,
                    "total_grupos_duplicatas": len(duplicatas_estruturadas),
                    "total_duplicatas_encontradas": sum(len(grupo["duplicatas_encontradas"]) for grupo in duplicatas_estruturadas),
                    "grupos_duplicatas": duplicatas_estruturadas
                }
                
                # Exibir conforme formato escolhido
                if formato_saida == "JSON Puro":
                    exibir_json_puro(resultados_finais["duplicidade"], "Duplicidade")
                else:
                    exibir_interface_visual(duplicatas, "Duplicidade")
                    
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
                
                # Preparar dados estruturados para JSON
                analise_textura = {
                    "parametros": {
                        "limiar_naturalidade": limiar_naturalidade,
                        "tamanho_bloco": tamanho_bloco,
                        "threshold_lbp": threshold_lbp
                    },
                    "total_imagens_analisadas": len(resultados_textura),
                    "resultados": []
                }
                
                for resultado in resultados_textura:
                    item_estruturado = {
                        "arquivo": resultado["nome"],
                        "indice": resultado["indice"],
                        "score_naturalidade": resultado["score"],
                        "categoria": resultado["categoria"],
                        "descricao": resultado["descricao"],
                        "percentual_areas_suspeitas": round(resultado["percentual_suspeito"], 2),
                        "detalhes_analise": resultado["analysis_results"]
                    }
                    analise_textura["resultados"].append(item_estruturado)
                
                resultados_finais["manipulacao_ia"] = analise_textura
                
                # Exibir conforme formato escolhido
                if formato_saida == "JSON Puro":
                    exibir_json_puro(resultados_finais["manipulacao_ia"], "Manipulação por IA")
                else:
                    exibir_interface_visual(resultados_textura, "Manipulação por IA")
                    
            except Exception as e:
                st.error(f"Erro durante a análise de textura: {str(e)}")
        
        # Se é análise completa e formato JSON, mostrar tudo junto
        if modo_analise == "Análise Completa" and formato_saida == "JSON Puro":
            st.markdown("## 📊 Resultado Completo")
            resultado_completo = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "modo_analise": modo_analise,
                "total_imagens_processadas": len(nomes),
                "nomes_arquivos": nomes,
                **resultados_finais
            }
            exibir_json_puro(resultado_completo, "Análise Completa")

else:
    # Mostrar exemplo quando não há imagens carregadas
    st.info("Faça upload de imagens para começar a detecção de fraudes.")
    
    if st.button("🔍 Ver exemplo de JSON", key="ver_exemplo_json"):
        exemplo_json = {
            "timestamp": "2025-06-06 14:30:00",
            "modo_analise": "Análise Completa",
            "total_imagens_processadas": 2,
            "nomes_arquivos": ["exemplo1.jpg", "exemplo2.jpg"],
            "duplicidade": {
                "metodo_usado": "SSIM + SIFT",
                "limiar_similaridade": 0.5,
                "total_grupos_duplicatas": 1,
                "total_duplicatas_encontradas": 1,
                "grupos_duplicatas": [
                    {
                        "imagem_original": {
                            "indice": 0,
                            "nome": "exemplo1.jpg"
                        },
                        "duplicatas_encontradas": [
                            {
                                "indice": 1,
                                "nome": "exemplo2.jpg",
                                "similaridade": 0.85,
                                "similaridade_percentual": 85.0
                            }
                        ]
                    }
                ]
            },
            "manipulacao_ia": {
                "parametros": {
                    "limiar_naturalidade": 50,
                    "tamanho_bloco": 16,
                    "threshold_lbp": 0.35
                },
                "total_imagens_analisadas": 2,
                "resultados": [
                    {
                        "arquivo": "exemplo1.jpg",
                        "indice": 0,
                        "score_naturalidade": 75,
                        "categoria": "Textura natural",
                        "descricao": "Baixa chance de manipulação",
                        "percentual_areas_suspeitas": 5.2
                    },
                    {
                        "arquivo": "exemplo2.jpg",
                        "indice": 1,
                        "score_naturalidade": 35,
                        "categoria": "Alta chance de manipulação",
                        "descricao": "Textura artificial detectada",
                        "percentual_areas_suspeitas": 45.8
                    }
                ]
            }
        }
        
        st.subheader("📄 Exemplo de JSON de Saída")
        json_str = json.dumps(exemplo_json, indent=2, ensure_ascii=False)
        st.code(json_str, language='json')

# Rodapé
st.markdown("---")
st.markdown("### Como interpretar os resultados")

# Explicação sobre duplicidade
if modo_analise in ["Duplicidade", "Análise Completa"]:
    st.write("""
    **Análise de Duplicidade (SSIM + SIFT):**
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
    - **Score 0-45**: Alta probabilidade de manipulação por IA  
    - **Score 46-70**: Textura suspeita, requer verificação manual
    - **Score 71-100**: Textura natural, baixa probabilidade de manipulação
    """)

# Contato e informações
st.sidebar.markdown("---")
st.sidebar.info("""
**Desenvolvido para:** Mirror Glass
**Projeto:** Detecção de Fraudes em Imagens Automotivas
**Versão:** 1.2.0 (Junho/2025)
**Método Duplicidade:** SSIM + SIFT
""")
